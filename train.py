import torch
from utils import dice_scores


def train(
        model,
        device, 
        train_dataloader, 
        val_dataloader,
        loss_fn,
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
            predict_masks = model(images).argmax(dim=1)
            loss = loss_fn(predict_masks, true_masks)
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()
            dice_score = dice_scores(predict_masks, true_masks)
            print(f"In epoch {epoch}, batch {batch_idx}, The dice score is {dice_score}")
        
        # Evaluate at the end of a epoch
        # Evaluation loop at the end of each epoch
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        val_dice = 0.0
        num_batches = 0
        
        with torch.no_grad():  # Disable gradient computation
            for batch_idx, batch in enumerate(val_dataloader):
                images, true_masks = batch[0], batch[1]
                assert images.shape[1] == 1, "The input image is expected to be one channel"
                
                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                
                logits = model(images)
                loss = loss_fn(logits, true_masks)
                predict_masks = logits.argmax(dim=1)
                dice_score = dice_scores(predict_masks, true_masks)
                
                val_loss += loss.item()
                val_dice += dice_score
                num_batches += 1
                
        # Average validation metrics
        avg_val_loss = val_loss / num_batches
        avg_val_dice = val_dice / num_batches
        print(f"End of epoch {epoch}: Validation Loss = {avg_val_loss:.4f}, Validation Dice Score = {avg_val_dice:.4f}")

                