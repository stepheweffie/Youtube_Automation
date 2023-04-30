import pyttsx3

# Set up the text-to-speech engine
engine = pyttsx3.init()

# Set the voice properties
voices = engine.getProperty('voices')
counter = 1
for i, v in enumerate(voices):
    text = "Why did the tomato turn red? Because it saw    the salad     dressing!"
    engine.setProperty('rate', 160)
    engine.setProperty('pitch', 80)
    engine.setProperty('voice', voices[i].id)
    engine.say(f"I am Voice Number {'  '}{counter}")
    engine.say(text)
    counter += 1

if __name__ == "__main__":
    engine.runAndWait()





