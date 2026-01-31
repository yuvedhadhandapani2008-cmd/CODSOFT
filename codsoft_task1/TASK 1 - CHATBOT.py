# Enhanced Rule-Based Chatbot Program
print("Bot: Hello! I am a simple rule-based chatbot.")
print("Bot: You can ask me about greetings, my name, help, date, or exit.")
print("Bot: Type 'bye' to end the conversation.\n")

while True:
    user_input = input("You: ").lower()

    # Greeting
    if user_input in ["hi", "hello", "hey"]:
        print("Bot: Hello! Nice to meet you.")

    # How are you
    elif "how are you" in user_input:
        print("Bot: I am doing well, thank you for asking!")

    # Bot name
    elif "name" in user_input:
        print("Bot: I am a rule-based chatbot created using Python.")

    # Help
    elif "help" in user_input:
        print("Bot: I respond using predefined rules and keywords.")

    # Thanks
    elif "thank" in user_input:
        print("Bot: You're welcome! Happy to help.")

    # Asking about creator
    elif "who created you" in user_input or "who made you" in user_input:
        print("Bot: I was created by a student as part of a chatbot task.")

    # Asking about purpose
    elif "what can you do" in user_input:
        print("Bot: I can respond to basic questions using rule-based logic.")

    # Date / time (dummy response)
    elif "time" in user_input:
        from datetime import datetime
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(f"Bot: Current time is {current_time}")

    # Exit
    elif "bye" in user_input:
        print("Bot: Goodbye! Have a great day.")
        break

    # Default response
    else:
        print("Bot: Sorry, I didn't understand that. Please try again.")
 