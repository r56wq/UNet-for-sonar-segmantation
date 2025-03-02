import torch

def train(
        model,
        device, 
        train_dataloader, 
        val_dataloader,
        loss_fn,
        evaluator,
        optimizer,
        epochs
    ):
    model = model.to(device)
    for epoch in range(epochs):
        model.train()
        for batch_idx, batch in enumerate(train_dataloader):
            images, true_masks = batch[0], batch[1]
            assert images.shape[1] == 1, "The input image is" \
            "excepted to be one channel"
            images = images.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.long)
            digits = model(images)
            loss = loss_fn(digits, true_masks)
            dice_score = evaluator(digits, true_masks)
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()
            print(f"In epoch {epoch}, batch {batch_idx}, the dice loss is {loss}, the dice score is {dice_score}")

        # evaluate at the end of a epoch
        model.eval()
        val_dice_score = 0;
        with torch.no_grad():
            for batch in val_dataloader:
                images, true_masks = batch[0], batch[1]
                assert images.shape[1] == 1, "The input image is" \
                "excepted to be one channel"
                images = images.to(device, dtype=torch.float32)
                true_masks = true_masks.to(device, dtype=torch.long)
                digits = model(images)
                dice_score = evaluator(digits, true_masks)
                val_dice_score += dice_score

            #Compute the average dice_socre per image
            val_dice_score /= len(val_dataloader)
            print(f"In epoch {epoch}, the dice score is {val_dice_score}")

            

                