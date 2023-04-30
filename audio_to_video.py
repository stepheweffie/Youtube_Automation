import os
import d6tflow
import moviepy.editor as mp
import openai
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


class AudioTextTask(d6tflow.tasks.TaskJson):

    def run(self):
        text = {'text': 'text strings'}
        self.save(text)


class ArtTextToVisualArt(d6tflow.tasks.TaskPqPandas):

    def run(self):
        self.save()


class TextToArt(d6tflow.tasks.TaskPqPandas):

    def run(self):
        data = self.inputLoad()
        audio_text = AudioTextTask()
        prompt = f"Generate art from audio text: {audio_text}"
        response = openai.Completion.create(
            engine="davinci",
            prompt=prompt,
            max_tokens=2048,
            n=1,
            stop=None,
            temperature=0.5,
        )

        art_text = response.choices[0].text.strip()
        art = mp.TextClip(art_text, font="Arial", fontsize=50, color="white", bg_color="black")
        art.write_videofile("art.mp4", fps=24)
        self.save(data)


class CreateVideo(d6tflow.tasks.TaskData):

    def output(self):
        return d6tflow.targets.DataTarget()

    def requires(self):
        return TextToArt()

    def run(self):
        data = self.inputLoad()
        audio_video = mp.VideoFileClip("audio.mp4")
        art_video = mp.VideoFileClip("art.mp4").set_position(("center", "bottom"))
        final_video = mp.CompositeVideoClip([audio_video, art_video])
        final_video.write_videofile("final_video.mp4")
        self.save(data)


if __name__ == "__main__":
    d6tflow.run(CreateVideo())

