import torch
from tqdm import tqdm


def _rotate_first_to_last(tensor):
    """
    Roll the first submatrix of [n_views, batch_size, dim] tensor to the end.
    Args:
        tensor: [n_views, batch_size, dim]
    Returns:
        Rolled tensor with the same shape
    """
    first_matrix = tensor[0:1]  # [1, batch_size, dim]
    remaining_matrices = tensor[1:]

    rotated_tensor = torch.cat([remaining_matrices, first_matrix], dim=0)

    return rotated_tensor


# Clustering via Self-Supervised Diffusion
# https://arxiv.org/abs/2507.04283

def mv_generate(models, model_diff, x, args, time_pairs, disable=True):
    x = x.unsqueeze(2).expand(-1, -1, x.shape[0], -1).reshape(x.shape[0], x.shape[1], -1)
    z_t = torch.randn_like(x, device=x.device)
    self_cond = torch.zeros_like(z_t, device=x.device)

    zs_g = []
    eta = 1

    for time, time_next in tqdm(time_pairs, desc='sampling mv loop time step', disable=disable):
        time_cond = torch.full((x.shape[1],), time, device=x.device, dtype=torch.long)
        if args.pred_v:
            model_output = [models[i](z_t[i], x[i], time_cond, self_cond[i]) for i in range(x.shape[0])]
            z_start = [model_diff.predict_start_from_v(z_t[i], time_cond, model_output[i]) for i in range(x.shape[0])]
            z_start = torch.stack([torch.clamp(z_start[i], min=-1., max=1.) for i in range(x.shape[0])])
        else:
            z_start = [models[i](z_t[i], x[i], time_cond, self_cond[i]) for i in range(x.shape[0])]
            z_start = torch.stack([torch.clamp(z_start[i], min=-1., max=1.) for i in range(x.shape[0])])
        pred_noise = torch.stack(
            [model_diff.predict_noise_from_start(z_t[i], time_cond, z_start[i]) for i in range(x.shape[0])]
        )

        if time_next < 0:
            zs_g.append(z_start)
            continue

        alpha = model_diff.alphas_cumprod[time]
        alpha_next = model_diff.alphas_cumprod[time_next]

        sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = (1 - alpha_next - sigma ** 2).sqrt()

        noise = torch.randn_like(z_t) * args.decoding_rescaling_factor

        z_t = z_start * alpha_next.sqrt() + \
              c * pred_noise + \
              sigma * noise
        zs_g.append(z_t)

        self_cond = _rotate_first_to_last(z_start)
    return z_t


def con_generate(model, model_diff, x, args, time_pairs, disable=True):
    x = x.permute(1, 0, 2).reshape(x.shape[1], -1)
    z_t = torch.randn(x.shape[0], x.shape[-1], device=x.device) * args.decoding_rescaling_factor
    self_cond = torch.zeros_like(z_t, device=x.device)

    zs_g = []
    eta = 1

    for time, time_next in tqdm(time_pairs, desc='sampling con loop time step', disable=disable):
        time_cond = torch.full((x.shape[0],), time, device=x.device, dtype=torch.long)
        if args.pred_v:
            model_output = model(z_t, x, time_cond, self_cond)
            z_start = model_diff.predict_start_from_v(z_t, time_cond, model_output)
            z_start = torch.clamp(z_start, min=-1., max=1.)
        else:
            z_start = model(z_t, x, time_cond, self_cond)
            z_start = torch.clamp(z_start, min=-1., max=1.)
        pred_noise = model_diff.predict_noise_from_start(z_t, time_cond, z_start)

        if time_next < 0:
            zs_g.append(z_start)
            continue

        alpha = model_diff.alphas_cumprod[time]
        alpha_next = model_diff.alphas_cumprod[time_next]

        sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = (1 - alpha_next - sigma ** 2).sqrt()

        noise = torch.randn_like(z_t) * args.decoding_rescaling_factor

        z_t = z_start * alpha_next.sqrt() + \
              c * pred_noise + \
              sigma * noise
        zs_g.append(z_t)

        self_cond = z_start
    return z_t


def train_iter_mv(models, model_diff, x, args, time):
    x = x.unsqueeze(2).expand(-1, -1, x.shape[0], -1).reshape(x.shape[0], x.shape[1], -1)
    noise = torch.randn_like(x) * args.decoding_rescaling_factor
    z_t = model_diff.q_sample(x, time, noise).type_as(x)
    self_cond = torch.zeros_like(x)
    if args.pred_v:
        pred_v = torch.stack([models[i](z_t[i], x[i], time, self_cond[i]) for i in range(x.shape[0])])
        z_0 = model_diff.predict_start_from_v(z_t, time, pred_v)
    else:
        z_0 = torch.stack([models[i](z_t[i], x[i], time, self_cond[i]) for i in range(x.shape[0])])
    return z_0


def train_iter_con(model, model_diff, x, args, time):
    x = x.permute(1, 0, 2).reshape(x.shape[1], -1)
    noise = torch.randn_like(x) * args.decoding_rescaling_factor
    z_t = model_diff.q_sample(x, time, noise).type_as(x)
    self_cond = torch.zeros_like(x)
    if args.pred_v:
        pred_v = model(z_t, x, time, self_cond)
        z_0 = model_diff.predict_start_from_v(z_t, time, pred_v)
    else:
        z_0 = model(z_t, x, time, self_cond)
    return z_0

