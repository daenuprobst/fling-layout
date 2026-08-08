import torch


def newton_schulz(grad, steps=5, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    x = grad.float()
    flip = x.shape[0] > x.shape[1]

    if flip:
        x = x.T

    x = x / (x.norm() + eps)

    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * gram @ gram) @ x

    return x.T if flip else x


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        groups,
        lr,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        betas=(0.9, 0.999),
        eps=1e-8,
    ):
        super().__init__(
            groups,
            dict(
                lr=lr,
                lr_mult=1.0,
                momentum=momentum,
                nesterov=nesterov,
                ns_steps=ns_steps,
                betas=betas,
                eps=eps,
                muon=False,
            ),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            # the training loops anneal by assigning lr into every group, so the per-group rate
            # lives in lr_mult and is applied here
            lr = group["lr"] * group.get("lr_mult", 1.0)

            for param in group["params"]:
                if param.grad is None:
                    continue

                state = self.state[param]

                if group["muon"]:
                    buf = state.setdefault("buf", torch.zeros_like(param))
                    buf.mul_(group["momentum"]).add_(param.grad)
                    upd = (
                        param.grad.add(buf, alpha=group["momentum"])
                        if group["nesterov"]
                        else buf
                    )
                    upd = newton_schulz(upd, group["ns_steps"])

                    # so a wide layer and a tall one move comparably in the norm the update is
                    # measured in
                    scale = max(1.0, param.shape[0] / param.shape[1]) ** 0.5
                    param.add_(upd.to(param.dtype), alpha=-lr * scale)
                    continue

                exp_avg = state.setdefault("exp_avg", torch.zeros_like(param))
                exp_sq = state.setdefault("exp_sq", torch.zeros_like(param))

                state["t"] = state.get("t", 0) + 1
                beta1, beta2 = group["betas"]

                exp_avg.mul_(beta1).add_(param.grad, alpha=1 - beta1)
                exp_sq.mul_(beta2).addcmul_(param.grad, param.grad, value=1 - beta2)

                bc1 = 1 - beta1 ** state["t"]
                bc2 = 1 - beta2 ** state["t"]

                denom = (exp_sq / bc2).sqrt_().add_(group["eps"])
                param.addcdiv_(exp_avg / bc1, denom, value=-lr)

        return loss


def build(net, lr, extra=(), ratio=2.0, min_dim=8):
    # `extra` holds parameters outside the network (a per-node code table). They are passed
    # explicitly rather than filtered by shape, since a code table can look like a hidden layer
    out = getattr(net, "out", None)
    excluded = set(id(p) for p in (out.parameters() if out is not None else []))

    layers, rest = [], list(extra)

    for param in net.parameters():
        if not param.requires_grad:
            continue

        if (
            param.ndim == 2
            and min(param.shape) >= min_dim
            and id(param) not in excluded
        ):
            layers.append(param)
        else:
            rest.append(param)

    groups = []

    if layers:
        groups.append(dict(params=layers, muon=True, lr_mult=ratio))

    if rest:
        groups.append(dict(params=rest, muon=False, lr_mult=1.0))

    return Muon(groups, lr=lr)
