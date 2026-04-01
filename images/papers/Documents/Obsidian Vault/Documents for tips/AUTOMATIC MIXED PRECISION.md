[`torch.amp`](https://pytorch.org/docs/stable/amp.html#module-torch.amp "torch.amp") provides convenience methods for mixed precision, where some operations use the `torch.float32` (`float`) datatype and other operations use lower precision floating point datatype (`lower_precision_fp`): `torch.float16` (`half`) or `torch.bfloat16`. Some ops, like linear layers and convolutions, are much faster in `lower_precision_fp`. Other ops, like reductions, often require the dynamic range of `float32`. Mixed precision tries to match each op to its appropriate datatype.

[`autocast`](https://pytorch.org/docs/stable/amp.html#torch.autocast "torch.autocast") should wrap only the forward pass(es) of your network, including the loss computation(s). Backward passes under autocast are not recommended. Backward ops run in the same type that autocast used for corresponding forward ops.

Example for CUDA Devices:

	# Creates model and optimizer in default precision
	model = Net().cuda()
		optimizer = optim.SGD(model.parameters(), ...)

	for input, target in data:
	    optimizer.zero_grad()

    # Enables autocasting for the forward pass (model + loss)
    with autocast():
        output = model(input)
        loss = loss_fn(output, target)

    # Exits the context manager before backward()
    loss.backward()
    optimizer.step()

[`autocast`](https://pytorch.org/docs/stable/amp.html#torch.autocast "torch.autocast") can also be used as a decorator, e.g., on the `forward` method of your model:

	class AutocastModel(nn.Module):
    ...
	    @autocast()
	    def forward(self, input):
	        ...
