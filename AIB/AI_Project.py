#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys

assert sys.version_info >= (3, 7)


# <p>In the text above I inted to check whether the Python version being used is 3.7 or greater.</p>

# <hr>

# <h1><center><u>A Project on Human-AI Collaboration - With a Focus on Historic Documents</u></center></h1>
# <b><h3>Student ID: 2661047</h3></b>
# <b><a href="https://github.com/2661047/AI_Project.git">Github URL</a></b><p> or if you aren't able to select it, it is below.</p>
# <p>https://github.com/2661047/AI_Project.git</p>

# <hr>

# <p>This project aims to use artificial intelligence (AI) alongside a historic document to increase accessibility and engagement with the past. The document of focus in this project is a travel journal that dates from the 19th century – 1867-1868. The journal is titled “Journal of a continental tour” (UofG, n.d.) and was acquired by the University of Glasgow in the 1950s. The journal contains handwritten text providing descriptions of the authors travels around Europe. Throughout the journal, the author provides illustrations and photos of different parts of their journey. These illustrations provide visual context on the people and scenery that surrounded the author through their journey.
# <br>
# <br>
# This travel journal is stored in the Archives and Special Collections department of the University of Glasgow, this allowed me to be hands on with the journal and take photos to better my analysis of the manuscript. Provided with several historic documents and manuscripts, I selected the journal due to it containing illustrations alongside descriptive text. The aim of my project is to use AI to increase accessibility to historic documents and increase engagement with historic documents. Firstly, I will explain how AI can be used for image generation. Displaying images alongside the illustrations within the journal presents greater visual appeal and context that can attract individuals to engage with the document. These images aim to add to the journal, rather than retract the importance and detail of the original illustrations. By attracting people to large, bold images they are more likely to read about this individual’s journey in the 19th century. The images created using AI may also be more distinct and significant to those visually impaired. If this document was to be an example of an artefact in a museum, attention-grabbing images become differentiated to their surroundings. For an individual who may be visually impaired, the colours of the images may still be seen and as they engage with the artefact, AI may also provide audio. Audio created using AI will provide engaging speech of the writings within the journal. The second part of this project will present how AI is able to convert text to audio. In doing so, this will also encourage engagement with those visually impaired and those looking to interact with the images whilst listening to the respective section of the journal. The final part of this project will present how AI can be used to convert audio to text. By providing subtitles along with the audio those who are deaf, or have hearing issues, may still be able to understand the contents of the journal.
# <br>
# <br>
# Overall, the project aims to demonstrate the wide use of AI and how it can be used with historic documents to further engagement and accessibility. Using AI generated images to make the manuscript more appealing, AI generated audio to increase access to the document, and AI generated text from the audio to further encourage accessibility. I will be using Hugging Face to assist in finding code, and also to assist in bettering my understanding of how the code operates.
# 
# </p>

# <hr>

# <h2><u>Image Generation</u></h2>

# <p>I began by trying to create my own code or implement code made by someone else found on HuggingFace. However, with every attempt this was not possible as my device would not be able to run the code. Therefore, for this project I used DALL-E 3 for image generation. DALL-E 3 is image generation software created by OpenAI. This third version of DALL-E produces images “that are not only more visually striking but also crisper in detail” (OpenAI, 2023).
#     <br>
#     <br>
# DALL-E uses a type of neural network architecture known as transformers. The use of transformers is evident in the name of ChatGPT, where DALL-E 3 is accessible. GPT stands for Generative Pre-Trained Transformer (Lighthouse Guild, n.d.). Transformers were created by a group in 2017 to improve the performance of language models in the paper “Attention Is All You Need” (Ashish Vaswani, 2017).
#     <br>
#     <br>
# As I provide a text input, the transformer model processes this and breaks it up into “a single stream of data” (Abideen, 2023). A decoder used by DALL-E is the Discrete Variational Autoencoder (dVAE) and is used to categorise aspects of the input, calculate probabilities, and make information “interpretable and valuable for various applications” (Abideen, 2023). When an input is given, a transformer processes this information which is then used with the decoder to produce a new image (Abideen, 2023). DALL-E uses images from Wikipedia and YFCC100M++ to produce these images (Abideen, 2023).
# 
# </p>

# <p>I provided ChatGPT with the following text – 
#     <br>
#     <br>
# <b>create an image that shows an image of the man with a moustache, as described here - "Before leaving the Paris station we were gratified by the sight of an interesting individual, accompanied by some ladies – whose physique was after this fashion. The moustache which is not in the slightest degree exaggerated was of a sandy grey"</b>
#     <br>
#     <br>
# This is text taken from the travel journal, describing an individual with a large moustache. An illustration of this man is provided below his writing.
# 
# </p>

# <div>
# <img src="DALL-E_Moustache.jpg" width="300"/>
# </div>

# <p>Above is the output provided by DALL-E 3. The result is a clear, engaging image that depicts a man of the description. However, I wanted to provide the original illustration to provide further information to better the results. Below is the original illustration found in the travel journal.</p>

# <div>
# <img src="illustration_moustache.jpg" width="300"/>
# </div>

# <p>The result, after providing this source image, can be seen bellow. The image is closer to the original, as the gentleman is wearing a top hat. This result has made the image appear more as photograph, by putting a filter over the image. This may have the result of making it more engaging, however if we want it to have more colour we can ask DALL-E 3 to produce a more vibrant image.</p>

# <div>
# <img src="DALL-E_Moustache_2.jpg" width="300"/>
# </div>

# <p>Below is the output provided by DALL-E 3 after providing the input – 
#     <br>
#     <br>
# <b>“can you make the image more vibrant with colour, but still aligned with the original description”</b></p>

# <div>
# <img src="DALL-E_Moustache_3.jpg" width="300"/>
# </div>

# <p>Overall, it can be said that these images have the result of quickly creating bold and vibrant images. The ease in being able to alter and change the results allows the user to customise the image until it is perfected. These images used alongside the journal may allow for it to become more engaging and eye catching.</p>

# <p>Below is another examples of images produced by DALL-E alongside the original image and the writing in the journal.</p>

# <p><b>create an image that follows this description from a travel journey in the 1800s - 
#     <br>
#     <br>
# There was a couple opposite to use – evidently newly married – who attracted a large share of our attention. Milady at the best was not good looking (which the bridegroom was) and what was at first a great paleness settled into a thorough greenness of complexion which must have been deeply interesting to her amiable companion. She lay back at last in his arms during the remainder of the passage while he contemplated her with a very rueful expression</b></p>

# <div>
# <img src="DALL-E_Lady_on_boat.jpg" width="300"/>
# </div>
# <div>
# <img src="Lady_on_boat.jpg" width="300"/>
# </div>

# <hr>

# <h2><u>Text to Audio</u></h2>

# <p>My following prospect for this project is to be able to use AI to convert written text to audio. An electronic transcription of the journal was made available by Bob McLean. By using this, it can be processed through AI to create an audible transcription. By generating an audio output, this allows the journal, and other written artefacts, to become more widely accessible. Many museums don’t offer services towards the blind, at a blind museum in Israel, visitors “felt that because of their physical appearance, the staff approached them as if they were “mentally retarded.” (Poria, Reichel, and Brandt, 2008, p. 123, as cited in Montsho, 2022). By providing audio descriptions, it provides those who are blind the opportunity to independently go through the Museum and understand the artefacts within. I believe it would be crucial to furthering accessibility, whilst also allowing other users a more engaging experience.
#     <br>
#     <br>
# The AI I used was coqui/XTTS-v2 that I found on Hugging Face on the public libraries and datasets. Once again, I was unable to run the code required for this to operate on Jupyter. However, I will provide a description of how the code works.
# </p>

# <p>The following is the code provided by coqui/XTTS-v2 to generate text-to-speech.</p>
# <br>
# <code>from TTS.api import TTS</code>
# <br>
# <p>This line of code imports the Text-to-Speech(TTS) module required.</p>
# <br>
# <br>
# <code>tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)</code>
# <br>
# <p>This code ensures that the TTS model is used, as well as stating that the GPU should be used.</p>
# <br>
# <br>
# <code>tts.tts_to_file(text="It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",</code>
# <br>
#     <code>file_path="output.wav",</code>
#     <br>
#     <code>speaker_wav="/path/to/target/speaker.wav",</code>
#     <br>
#     <code>language="en")</code>
# <br>
# <p>This code creates an audio file after text input is provided. In "text" an input is given by the user, however, the transcription by Bob McLean could be used in order to generate audio of the writing within the journal. The "file_path" saves the audio. The "speaker_wav" obtains the speakers voice to be used. A voice of a male gentleman with Victorian tonalities may be used to create a sense that the writer of the journal is speaking to the listener. Finally, the "language="en"" specifies the language of the code. coqui/XTTS-v2 supports 17 languages (Cocqui, n.d.). This means that this journal has the possibility of becoming more widely accessible to those who do not speak english, it provides the option to listen to the contents of the journal and engage with history.</p>

# <p>Other code provided in the coqui/XTTS-v2 is the following</p>
# <br>
# <code>from TTS.tts.configs.xtts_config import XttsConfig</code>
# <br>
# <p>The above code imports "XttsConfig".</p>
# <br>
# <br>
# <code>from TTS.tts.models.xtts import Xtts</code>
# <br>
# <p>This code imports "Xtts".</p>
# <br>
# <br>
# <code>config = XttsConfig()</code>
# <br>
# <p>This code configures the necessary data for the "Xtts" model.</p>
# <br>
# <br>
# <code>config.load_json("/path/to/xtts/config.json")</code>
# <br>
# <p>This loads the configuration from the "json" file.</p>
# <br>
# <br>
# <code>model = Xtts.init_from_config(config)</code>
# <br>
# <p>This uses the configuration to begin the Xtts model.</p>
# <br>
# <br>
# <code>model.load_checkpoint(config, checkpoint_dir="/path/to/xtts/", eval=True)</code>
# <br>
# <p>This loads the model from the directory, and also sets the evaluation to "True".</p>
# <br>
# <br>
# <code>model.cuda()</code>
# <br>
# <p>This moves the model to the GPU. This is where I was unable to run the code myself.</p>
# <br>
# <br>
# <code>outputs = model.synthesize(</code>
# <br>
#     <code>"It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
#     config,</code>
#     <br>
#     <code>speaker_wav="/data/TTS-public/_refclips/3.wav",</code>
#     <br>
#     <code>gpt_cond_len=3,</code>
#     <br>
#     <code>language="en",</code>
#     <br>
# <code>)</code>
# </code>
# <br>
# <p>Here an input of text is given, "It took me quite a long time to develop a voice and now that I have it I am not going to be silent." is the text provided on the code on Hugging Face. Though, this is interchangable and can be altered to be writing from the journal or other historic documents. The "speaker_wav" obtains the speakers voice used for the speech. "gpt_cond_len=3," sets the length of text for GPT model. Finally, "language="en"," sets the 1 of 17 languages, in this circumstance it is selected to be English. </p>
# <br>
# <b><a href="https://huggingface.co/spaces/coqui/xtts">Selecting here will take you to Hugging Face where you can test the capabilities of the TTS.</a></b>
# <br>
# <br>
# <p>AI is used throughout this code in order to generate text-to-speech. The Xtts model is a neural network model. By loading the pretrained model, it then gathers data from the text input and then synthesises speech,</p>

# <hr>

# <h2><u>Audio to Text</u></h2>

# <p>Audio to text can be a beneficial tool in aiding accessibility and engagement of historic artefacts. The travel journal was a handwritten book, however, the handwriting is difficult to read for modern readers. This prompted Bob McLean to create an elligible transcription. This transcription could be used to display along with the journal to allow individuals to read it. Though, I would recommend using AI to translate the audio into text. In doing so, it provides a live transcription of what is being said. Increasing accessibility for those who are deaf, as well as encouraging engagement as individuals will be able to read in live time as they listen to the contents of the journal. Sign language is often unavailable at museums and this can limit accessibility, “The deaf often find themselves helpless as museum staff do not understand sign language.” (Montsho, 2022). The model used below was found on Hugging Face, it is called openai/whisper-large-v3. I will expand on the code below where its results can be seen.</p>

# <div>
# <img src="Whisper_AI.jpg" width="600"/>
# </div>
# <p>Above is an image sourced by Whisper, providing a visual depiciton on how the data is trained and formatted.</p>
# <br>
# <b><a href="https://github.com/openai/whisper">It can be seen on their GitHub here.</a></b>

# <p>The code below installs Whisper.</p>

# <code>pip install -U openai-whisper</code>
# <br>
# <p>I have had to turn this code into HTML as a requirements.txt was unable to be produced as it created a syntax error. An HTML copy of this project displays the output.</p>

# <p>The code below "the latest commit from this repository, along with its Python dependencies"(Whisper, n.d.) </p>

# <code>pip install git+https://github.com/openai/whisper.git</code>
# <br>
# <p>I have had to turn this code into HTML as a requirements.txt was unable to be produced as it created a syntax error. An HTML copy of this project displays the output.</p>

# <p>The follwoing code updates "the package to the latest version of this repository"(Whisper, n.d.)</p>

# <code>pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git</code>
# <br>
# <p>I have had to turn this code into HTML as a requirements.txt was unable to be produced as it created a syntax error. An HTML copy of this project displays the output.</p>

# <p>The command-line tool, ffmpeg, is then installed on the system. As I am using a Mac opererating system the code used is</p>
# <br>
# <code>brew install ffmpeg</code>
# <br>
# <p>By following this <b><a href="https://github.com/openai/whisper">link</a></b> , you will be able to see what code is required for your operating system.</p>

# <p>This code installs rust.</p>

# <code>pip install setuptools-rust</code>
# <br>
# <p>I have had to turn this code into HTML as a requirements.txt was unable to be produced as it created a syntax error. An HTML copy of this project displays the output.</p>

# <p>This code imports Whisper, then loads the model. The model is then used to transcribe the audio, "audio.mp3". In this circumstance I downloaded audio of an individual saying "Please choose one of the following options". The transcription is then outputted using "print". Here it can be seen to accurately generate an output.</p>

# In[43]:


import whisper

model = whisper.load_model("base")
result = model.transcribe("audio.mp3")
print(result["text"])


# <p>The code above that allows for the Whisper model by OpenAI to be used does not directly use AI. However, the Whisper model is built on the foundation of training large datasets using AI. Once the model is downloaded, called "base" in the aboce code, the code then processes the audio file to output a result.</p>

# <hr>

# <h2><u>Summary</u></h2>

# <p>In conclusion, it is evident that access to historic documents and manuscripts is lacking. Though, use of AI, as seen above, may allow for more individuals to interact with these manuscripts. Assisting the blind and deaf in the understanding of the past. Easily applicable, the use of these models are possible to be applied to every museum and space where necessary. Going beyond this, the use of these models further encourage engagement between individuals and artefacts. By providing more vibrant and modern image depictions alongside the original illustrations, individuals may feel more attracted towards the artefact and encouraged to learn more about it. It may also be allow for those who are visually impaired to understand some of the depictions within the journal through colours. Following this, the use of text to audio allows those who are blind or visually impaired to further their engagement with the artefact. Individuals may also choose to interact with the manuscript through audio, whilst the respective image is displayed. Finally, the audio to text allows for a live transcription of what is being said. Those who are deaf will then be able to interact with the journal as the writing within the journal is difficult to read.
#     <br>
#     <br>
# Overall, it can be seen that through the use of AI, users will be more likely to engage with the past – increasing accessibility and engagement.
# </p>

# <h3><u>Works Cited</u></h3>

# <p>Abideen, Z. u., 2023. How OpenAI's DALL-E works?. [Online] 
# Available at: <b><a href="https://medium.com/@zaiinn440/how-openais-dall-e-works-da24ac6c12fa#:~:text=DALL%2DE%20uses%20the%20transformer,short%2Drange%20dependencies%20between%20pixels">https://medium.com/@zaiinn440/how-openais-dall-e-works-da24ac6c12fa#:~:text=DALL%2DE%20uses%20the%20transformer,short%2Drange%20dependencies%20between%20pixels</a></b>
# [Accessed April 2024].
#     <br>
#     <br>
# Ashish Vaswani, N. S. N. P. J. U. L. J. A. N. G. L. K. I. P., 2017. Attention Is All You Need. [Online] 
# Available at: <b><a href="https://arxiv.org/abs/1706.03762">https://arxiv.org/abs/1706.03762</a></b>
# [Accessed April 2024].
#     <br>
#     <br>
# Cocqui, n.d. cocqui/XTTS-v2. [Online] 
# Available at: <b><a href="https://huggingface.co/coqui/XTTS-v2">https://huggingface.co/coqui/XTTS-v2</a></b>
# [Accessed April 2024].
#     <br>
#     <br>
# Lighthouse Guild, n.d. What is Chat GPT. [Online] 
# Available at: <b><a href="https://lighthouseguild.org/what-is-chat-gpt/">https://lighthouseguild.org/what-is-chat-gpt/</a></b>
# [Accessed April 2024].
#     <br>
#     <br>
# Montsho, G., 2022. Making Museums Accessible to Those With Disabilities. [Online] 
# Available at: <b><a href="https://www.museumnext.com/article/making-museums-accessible-to-those-with-disabilities/">https://www.museumnext.com/article/making-museums-accessible-to-those-with-disabilities/</a></b>
# [Accessed April 2024].
#     <br>
#     <br>
# OpenAI, 2023. DALL·E 3 is now available in ChatGPT Plus and Enterprise. [Online] 
# Available at: <b><a href="https://openai.com/blog/dall-e-3-is-now-available-in-chatgpt-plus-and-enterprise">https://openai.com/blog/dall-e-3-is-now-available-in-chatgpt-plus-and-enterprise</a></b>
# [Accessed April 2024].
#     <br>
#     <br>
# UofG, n.d. Collections. [Online] 
# Available at:  <b><a href="https://www.gla.ac.uk/collections/#/details?irn=250262&catType=C&referrer=/results&q=ms+gen+13">https://www.gla.ac.uk/collections/#/details?irn=250262&catType=C&referrer=/results&q=ms+gen+13</a></b>
# [Accessed April 2024].
#     <br>
#     <br>
# Whisper, n.d. openai.whisper. [Online] 
# Available at:  <b><a href="https://github.com/openai/whisper">https://github.com/openai/whisper</a></b>
# [Accessed April 2024].
# </p>
# 

# In[ ]:




