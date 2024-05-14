# Stego-Tool
This project is a steganography tool that hides a message inside of a video. This message is password protected and can be decrypted with the same executable.

## Usage
First of all, clone this repository in your local files, then execute pip3 install -r requirements.txt to satisfy the possible requirements thta were not satisfied before:

```bash
pip3 install -r requirements.txt
```

After installing the packages everything is ready to use:

You will need:
- A picture in the file format .png that will be used for the background in the output video.
- An audio file that must be in format .wav that will be the sound of the video.

When inputted this files into the program, we will be asked for a password to encode the message. Later, if we have not yet inputted the filename of the output video, we will be prompted to introduce it.

### Examples:
For encoding a message into a video
```bash
python3 main.py -e -i \<IMAGE\> -a \<AUDIO\> -v \<OUTPUT VIDEO NAME\> 
```

For decoding the video
```bash
python3 main.py -d -i \<IMAGE\> -a \<AUDIO\> -v \<OUTPUT VIDEO NAME\> 
```