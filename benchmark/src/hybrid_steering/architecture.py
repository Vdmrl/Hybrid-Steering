"""Small, explicit adapters for the decoder families used in experiments."""


def decoder_layers(model):
    if model.config.model_type == "falcon_h1":
        return model.model.layers
    return model.model.language_model.layers


def recurrent_family(model) -> str:
    return "mamba" if model.config.model_type == "falcon_h1" else "gdn"


def restore_sequence_axis(module, args, output):
    """Work around Falcon RMSNormGated squeezing length-1 decode outputs.

    Without this, [B,D] + [B,1,D] broadcasts to [B,B,D] for B>1.
    Already-correct Transformers versions pass through unchanged.
    """
    if args[0].ndim == 3 and output.ndim == 2 and args[0].shape[1] == 1:
        return output.unsqueeze(1)
    return output


def adapt_model(model) -> None:
    if recurrent_family(model) == "mamba":
        for layer in decoder_layers(model):
            if layer.mamba.mamba_rms_norm:
                layer.mamba.norm.register_forward_hook(restore_sequence_axis)
