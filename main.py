from face_recognition_app import FaceRecognitionSystem

def main():
    face_system = FaceRecognitionSystem()
    
    while True:
        print("\nFace Recognition System")
        print("1. Register new face")
        print("2. Start face recognition")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            name = input("Enter the name for the new face: ")
            face_system.register_new_face(name)
        elif choice == '2':
            face_system.start_recognition()
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 