import cv2


# image_path = 'res.png'
# image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

# mask = image[..., :3].sum(-1) > 100
# breakpoint()

image0 = cv2.imread('res1.png', cv2.IMREAD_UNCHANGED)
image0 = cv2.resize(image0, (800, 800), dst=None, fx=None, fy=None, interpolation=cv2.INTER_LINEAR)
breakpoint()
cv2.imwrite('res1_.png', image0)
