module UNet

using Flux

struct UNetModel
    downsample_layers::Array{Chain}
    upsample_layers::Array{Chain}
    skip_connections::Array{Array{Chain}}
    final_conv::Chain
end

function UNetModel(input_shape::Tuple, num_classes::Int, num_filters::Int)
    downsample_layers = [
    ]
    upsample_layers = [
    ]
    skip_connections = [
       

    ]
    final_conv = Chain(x -> reshape(x, input_shape..., 1),
   
    return UNetModel(downsample_layers, upsample_layers, skip_connections, final_conv)
end

function (m::UNetModel)(x)
end

end # module
