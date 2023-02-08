function train_unet(model, train_data, val_data, opts)
    # Initialize the loss function and the optimizer
    criterion = BinaryCrossEntropy()
    optimizer = ADAM(params(model), opts.learning_rate)

    # Begin training loop
    for epoch in 1:opts.epochs
        train_loss = 0.0
        train_count = 0

        # Iterate over the training data
        for (inputs, labels) in train_data
            # Zero out gradients from previous iteration
            optimizer.zero_grad()

            # Make predictions using the model
            outputs = model(inputs)

            # Calculate the loss
            loss = criterion(outputs, labels)

            # Backpropagate to calculate gradients
            loss.backward()

            # Update model parameters
            optimizer.step()

            # Add to running total of training loss
            train_loss += loss.item()
            train_count += 1
        end

        # Calculate the average training loss for this epoch
        avg_train_loss = train_loss / train_count

        # Evaluate the model on the validation data
        val_loss = evaluate_unet(model, val_data, criterion)

        # Print training and validation loss for this epoch
        println("Epoch: $(epoch), Train Loss: $(avg_train_loss), Val Loss: $(val_loss)")
    end
end

function evaluate_unet(model, val_data, criterion)
    val_loss = 0.0
    val_count = 0

    # Iterate over the validation data
    for (inputs, labels) in val_data
        # Make predictions using the model
        outputs = model(inputs)

        # Calculate the loss
        loss = criterion(outputs, labels)

        # Add to running total of validation loss
        val_loss += loss.item()
        val_count += 1
    end

    # Calculate the average validation loss
    avg_val_loss = val_loss / val_count

    return avg_val_loss
end
