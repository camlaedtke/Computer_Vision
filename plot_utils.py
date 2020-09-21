import numpy as np
import matplotlib.pyplot as plt


def plot_history(results):
    
    plt.figure(figsize=(15,7))

    plt.subplot(1,3,1)  
    plt.plot(results.history['loss'], 'r', label='Training loss')
    plt.plot(results.history['val_loss'], 'b', label='Validation loss')
    plt.title('Log Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('log_loss', fontsize=16)
    # plt.ylim([0, 1])
    plt.legend()

    plt.subplot(1,3,2)
    plt.plot(results.history['accuracy'], 'r', label='Training accuracy')
    plt.plot(results.history['val_accuracy'], 'b', label='Validation accuracy')
    plt.title('Accuracy', fontsize=16)
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    # plt.ylim([0, 1])
    plt.legend()

    plt.subplot(1,3,3)
    plt.plot(results.history['dice_coef'], 'r', label='Dice coefficient')
    plt.plot(results.history['val_dice_coef'], 'b', label='Validation dice coefficient')
    plt.title('Dice Coefficient', fontsize=16)
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Dice', fontsize=16)
    # plt.ylim([0, 1])
    plt.legend()
    plt.show()
    
    
def plot_dice_and_iou(trainId2label, n_classes, class_dice, class_iou):
    categories = [trainId2label[i].category for i in range(n_classes)]
    cmap = [color['color'] for color in plt.rcParams['axes.prop_cycle']]
    cat_colors = {
        'void': 'black',
        'flat': cmap[0],
        'construction': cmap[1],
        'object': cmap[2],
        'nature': cmap[3],
        'sky': cmap[4],
        'human': cmap[5],
        'vehicle': cmap[6]
    }
    colors = [cat_colors[category] for category in categories]

    names = [trainId2label[i].name for i in range(n_classes)]

    plt.style.use('ggplot')


    plt.figure(figsize=(15,20))

    plt.subplot(2,1,1)
    plt.barh(names, class_dice, color=colors)
    plt.xlabel("Dice Coefficient", fontsize=18)
    plt.ylabel("Class Name", fontsize=18)
    plt.title("Class Dice Scores", fontsize=22)
    plt.xlim([0, 1])

    plt.subplot(2,1,2)
    plt.barh(names, class_iou, color=colors)
    plt.xlabel("Intersection Over Union", fontsize=18)
    plt.ylabel("Class Name", fontsize=18)
    plt.title("Class IOU Scores", fontsize=22)
    plt.xlim([0, 1])
    plt.show()
    
    
    