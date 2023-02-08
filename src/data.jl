using Flux
using MLJ
using BSON
using Statistics
using MLDatasets

# Load the data from the specified directory
function load_data(path)
    # Use BSON library to load the data from a BSON file
    data = BSON.load(path)

    # Split the data into training and validation sets
    train_data, valid_data = splitobs(data, at = 0.8)

    return (train_data, valid_data)
end

# Preprocess the training and validation data
function preprocess_data(data)
    # Normalize the data using the mean and standard deviation
    data = normalize(data, MeanStd())

    # Convert the data to Flux's tensors
    data = (Flux.data(data[!, 1:end-1]), Flux.onehotbatch(data[!, end]))

    return data
end
