def pick_test_images():
    test_imgs = []
    for i in range(1, 51):
        name = '../Data/test_set_images/test_'+str(i)+'/test_' + str(i) + '.png'
        test_imgs.append(load_image(name))
    return test_imgs
        
test = pick_test_images()