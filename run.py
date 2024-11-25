import cv2
import argparse
from model import Model


def argument_parser():
    parser = argparse.ArgumentParser(description="Violence detection")
    parser.add_argument('--image-path', type=str,
                        default='D:/NCKH/violence-detection/data/fire.mp4',
                        help='path to your image or video')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argument_parser()
    model = Model()

    # Xử lý video
    if args.image_path.endswith('.mp4'):  # Kiểm tra nếu là video
        cap = cv2.VideoCapture(args.image_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {args.image_path}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Kết thúc video
            
            # Dự đoán nhãn cho từng frame
            label = model.predict(image=frame)['label']
            print('predicted label: ', label)
            
            # Hiển thị frame với nhãn
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Violence Detection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Nhấn 'q' để thoát
                break
        
        cap.release()
        cv2.destroyAllWindows()
    else:
        # Xử lý hình ảnh
        image = cv2.imread(args.image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot open image file: {args.image_path}")
                                                                        
        label = model.predict(image=image)['label']
        print('predicted label: ', label)
        cv2.imshow(label.title(), image)
        cv2.waitKey(0)
