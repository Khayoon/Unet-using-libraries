module Upsample

using Flux

# Function to perform upsampling on an input tensor
function upsample(input_tensor, scale_factor)
    # Define the upsampling layer
    upsample_layer = Flux.upsample(input_tensor, scale_factor)

    # Return the upsampled tensor
    return upsample_layer
end

end # module
