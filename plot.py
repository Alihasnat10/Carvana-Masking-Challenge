import matplotlib.pyplot as plt

def display(display_list,filename):
    plt.figure(figsize=(15, 15))

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.imshow(display_list[i])
        plt.axis('off')
    plt.savefig("test_images/outputs/"+filename)
    plt.show()

def plot_model_results(results):
    fig, axis = plt.subplots(1, 2, figsize=(20, 5))
    axis[0].plot(results.history["loss"], color='r', label = 'train loss')
    axis[0].plot(results.history["val_loss"], color='b', label = 'dev loss')
    axis[0].set_title('Loss Comparison')
    axis[0].legend()
    axis[1].plot(results.history["dice_coef"], color='r', label = 'train dice_coef')
    axis[1].plot(results.history["val_dice_coef"], color='b', label = 'dev dice_coef')
    axis[1].set_title('Accuracy Comparison')
    axis[1].legend()