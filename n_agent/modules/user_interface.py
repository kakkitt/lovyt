from modules.response_generator import ResponseGenerator

class UserInterface:
    def __init__(self):
        self.response_generator = ResponseGenerator()

    def run(self):
        print("AI 어시스턴트에 오신 것을 환영합니다!")
        print("질문을 입력하거나 'quit'을 입력하여 종료하세요.")
        
        while True:
            user_input = input("\n사용자: ")
            if user_input.lower() == 'quit':
                print("프로그램을 종료합니다. 감사합니다!")
                break
            
            response = self.response_generator.generate_response(user_input)
            print(f"\nAI 어시스턴트: {response}")

if __name__ == "__main__":
    ui = UserInterface()
    ui.run()