module Downsampler

using Flux

# Function to perform downsampling on an input tensor
function downsample(input_tensor, scale_factor)
    # Define the downsampling layer
    downsample_layer = Flux.downsample(input_tensor, scale_factor, mode="nearest")

    # Measure the efficacy and efficiency of the downsampler
    efficacy = Flux.Evaluator.accuracy(input_tensor, downsample_layer)
    efficiency = Flux.Evaluator.time(input_tensor, downsample_layer)

    # Return the downsampled tensor and measure of efficacy and efficiency
    return downsample_layer, efficacy, efficiency
end

end
