import cv2
import torch
import timm
from torchvision import transforms
from PIL import Image

# emotion
EMO_NUM_CLASSES   = 7
EMO_MODEL_PATH    = 'emotion_mobilevit.pth'
emo_classes       = ['angry','disgust','fear','happy','neutral','sad','surprise']

# race
RACE_NUM_CLASSES  = 5
RACE_MODEL_PATH   = 'race.pth'
race_classes      = ['Asian','Black','Indian','Others','White']

# gender
GENDER_NUM_CLASSES = 2
GENDER_MODEL_PATH  = 'gender.pth'
gender_classes     = ['female','male']

AGE_NUM_CLASSES   = 10
AGE_MODEL_PATH    = 'age.pth'
age_classes       = [f"{i*10}-{i*10+9}" for i in range(AGE_NUM_CLASSES)]

IMG_SIZE     = 224
RGB_SIZE     = 48
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CASCADE_PATH = r'C:\cv2data\haarcascade_frontalface_default.xml'

def build_model(num_classes, model_path, in_chans):
    model = timm.create_model(
        'mobilevit_s',
        pretrained=False,
        num_classes=num_classes,
        img_size=IMG_SIZE,
        in_chans=in_chans
    )
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

emo_model    = build_model(EMO_NUM_CLASSES,   EMO_MODEL_PATH,   in_chans=1)
race_model   = build_model(RACE_NUM_CLASSES,  RACE_MODEL_PATH,  in_chans=3)
gender_model = build_model(GENDER_NUM_CLASSES, GENDER_MODEL_PATH, in_chans=3)
age_model    = build_model(AGE_NUM_CLASSES,   AGE_MODEL_PATH,   in_chans=3)

emo_preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

rgb_preprocess = transforms.Compose([
    transforms.Resize((RGB_SIZE, RGB_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    raise IOError(f"無法加載 Cascade: {CASCADE_PATH}")

colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255)
]

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for idx, (x, y, w, h) in enumerate(faces):
                color = colors[idx % len(colors)]
                crop_bgr = frame[y:y+h, x:x+w]
                img_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                
                emo_input = emo_preprocess(Image.fromarray(img_rgb)) \
                                .unsqueeze(0).to(DEVICE)
                rgb_input = rgb_preprocess(Image.fromarray(img_rgb)) \
                                .unsqueeze(0).to(DEVICE)

                emo_label    = emo_classes[   emo_model(emo_input).argmax(1).item() ]
                race_label   = race_classes[  race_model(rgb_input).argmax(1).item() ]
                gender_label = gender_classes[gender_model(rgb_input).argmax(1).item() ]
                age_label    = age_classes[   age_model(rgb_input).argmax(1).item() ]

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                txt = f"{emo_label} | {race_label} | {gender_label} | {age_label}"
                cv2.putText(frame, txt, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.namedWindow('Emotion | Race | Gender | Age', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Emotion | Race | Gender | Age', 1280, 720)
            cv2.imshow('Emotion | Race | Gender | Age', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
