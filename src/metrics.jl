module Metrics
    using Flux
    
    export accuracy, dice_coefficient
    
    function accuracy(y_pred::Array, y_true::Array)
        n = size(y_pred, 1)
        acc = 0
        for i in 1:n
            acc += (y_pred[i] == y_true[i])
        end
        acc / n
    end
    
    function dice_coefficient(y_pred::Array, y_true::Array)
        # Flatten the arrays to 1-D arrays
        y_pred = reshape(y_pred, (size(y_pred, 1),))
        y_true = reshape(y_true, (size(y_true, 1),))
        numerator = 2 * sum(y_pred .* y_true)
        denominator = sum(y_pred) + sum(y_true)
        numerator / denominator
    end
    
end
