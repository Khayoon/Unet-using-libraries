module Config
    # Model hyperparameters
    const num_epochs = 100
    const learning_rate = 0.001
    const batch_size = 32
    const input_shape = (256, 256, 3)

    # Data file paths
    const training_data_path = "path/to/training/data"
    const validation_data_path = "path/to/validation/data"

    # Output directories for saved models and figures
    const model_dir = "path/to/models"
    const figure_dir = "path/to/figures"
end
