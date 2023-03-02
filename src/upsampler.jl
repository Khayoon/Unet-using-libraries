function upsample(input_tensor, scale_factor, dims=2)
    # Define the upsampling layer
    if dims == 2
        upsample_layer = Flux.upsample(input_tensor, scale_factor)
    else
        upsample_layer = Flux.upsample(input_tensor, (scale_factor...,))
    end

    # Add a Tanh activation function
    upsample_layer = tanh.(upsample_layer)

    # Add a DDim layer for dynamic upsampling
    ddim_layer = Flux.DDim(size(upsample_layer))
    upsampled_tensor = ddim_layer(upsample_layer, (size(input_tensor) ÷ scale_factor...,))

    # Add instance normalization
    upsampled_tensor = Flux.instance_norm(upsampled_tensor)

    # Add dropout regularization
    upsampled_tensor = Flux.dropout(upsampled_tensor, 0.2)

    # Return the upsampled tensor
    return upsampled_tensor
end
