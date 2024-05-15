import random
from argparse import *
import cv2
import os
import hashlib
import base64
from Crypto.Cipher import AES
from pydub import AudioSegment
from pydub.silence import detect_silence
from pydub.generators import WhiteNoise
from moviepy.editor import AudioFileClip, ImageClip, concatenate_videoclips, CompositeVideoClip, VideoFileClip
from shutil import rmtree
import traceback

# Function to check if the file extension is the correct one
def check_extension(file, file_type):
	if(file == ''):
		return False
	extension = os.path.splitext(file)[1][1:4]
	if(file_type == 'image' and extension == "png"):
		return True
	if(file_type == 'video' and extension in ['avi']):
		return True
	if(file_type == 'audio' and extension in ['wav']):
		return True
	return False

#############################
#           ENCODE          #
#############################

def get_img_coords(img, pixel, add):
	new_pixel = pixel + add
	
	new_width = new_pixel % img.shape[1]
	new_height = new_pixel // img.shape[1]

	# We check if the pixel is in range
	if(new_width >= img.shape[1] or new_height >= img.shape[0]):
		print('ERROR: The calculated positions are outside the image.')
		exit(1)

	return new_pixel, new_width, new_height

def add_noise(file:AudioSegment, duration, start):
	noise = WhiteNoise().to_audio_segment(duration=duration)
	silence = AudioSegment.silent(duration=start)
	change_in_dBFS = -80 - noise.dBFS
	final = noise.apply_gain(change_in_dBFS)
	return file.overlay(silence + final)

def analyze_silences(audio_filename, min_silence_len):
	file = AudioSegment.from_wav(audio_filename)
	file_edited = file

	output_file = "./temp/audio_edited.wav"

	# First check that our audio file does not have any absolute silence
	silence_range = detect_silence(file, 3, silence_thresh=-100000000)
	if len(silence_range) > 0:
		# If the audio has absolute silence, we need to add noise
		print('WARNING: Absolute silence found in audio file. Adding noise')
		for value in silence_range:
			start = value[0]
			end = value[1]
			duration = end - start
			file_edited = add_noise(file, duration, start)
		file_edited.export(output_file, format="wav")
	else:
		file_edited.export(output_file, format="wav")
	
	# Now we can check the ranges where the silence threshold is under -60 dBFS
	neccessary_silence = min_silence_len*5+40
	ranges = detect_silence(file, neccessary_silence, -60.0)
	if(len(ranges) == 0):
		print('ERROR: No silence found in audio file. Try with other audio')
		exit(1)
	return ranges, output_file

def decimalToBinary(decimal_number):
	return bin(decimal_number).replace("0b", "")

def encode_image(file, plaintext_password, message):
	while(check_extension(file, 'image') == False):
		file = input("Introduce the file of the image (must be .png): ")
	img = cv2.imread(file)
	
	try:
		orig_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	except:
		print("ERROR: Image encoding is not supported")
		rmtree("./temp")
		exit(1)

	cv2.imwrite("./temp/img_bgr.png", orig_img)

	current_width = random.randrange(1, img.shape[0] // 2, 1)
	current_height = random.randrange(1, img.shape[1] // 2, 1)
	initial_pixel_num = (current_height * img.shape[1] + current_width)

	while(plaintext_password == ''):
		plaintext_password = input('Introduce the password to encode: ')

	password = hashlib.sha256(plaintext_password.encode("utf-8")).digest()

	while(message == ''):
		message = input('Introduce the message to be encoded: ')

	cipher = AES.new(password, AES.MODE_EAX)
	nonce = cipher.nonce
	ciphertext, tag = cipher.encrypt_and_digest(message.encode("utf-8"))
	message = base64.b64encode(nonce).decode("utf-8")
	message += base64.b64encode(tag).decode("utf-8")
	message += base64.b64encode(ciphertext).decode("utf-8")
	message += chr(145) * 3 # End of the message

	# Check if the message fits in the image
	if((initial_pixel_num + len(message) * 256) >= (img.shape[0] * img.shape[1])):
		print('ERROR: The message entered is too large to store in this image')
		exit(1)

	# Hide the message in the image
	index = 0
	current_pixel = initial_pixel_num
	while(index < len(message)):
		pixel = img[current_height][current_width]
		pixel[2] = ord(message[index])
		jump = pixel[0] if pixel[0] != 0 else 100
		current_pixel, current_width, current_height = get_img_coords(img, current_pixel, jump)
		index += 1

	# Export the image with the message
	new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	out_file = "./temp/img_stego.png"
	cv2.imwrite(out_file, new_img)

	return initial_pixel_num, out_file

def encode_audio(audio_filename, number):
	while(check_extension(audio_filename, 'audio') == False):
		audio_filename = input("Introduce the filename of the audio (must be .wav): ")

	number = decimalToBinary(int(number))

	ranges, audio_filename = analyze_silences(audio_filename, len(number))

	file = AudioSegment.from_wav(audio_filename)

	random_range = random.randint(0, len(ranges)-1)
	length = len(str(number))
	final_audio = AudioSegment.silent(duration=length*5)
	streak = 1
	index = 0
	for i in range(length):
		if index >= length:
			break
		if number[index] == '0':
			index += 1
		else:
			for j in range(index+1, length):
				if number[j] == '1':
					streak += 1
				else:
					break	 
			start = index*5
			noise = WhiteNoise().to_audio_segment(duration=streak*5)
			change_in_dBFS = - 60 - noise.dBFS
			final = noise.apply_gain(change_in_dBFS)
			final_audio = final_audio.overlay(final, position=start)
			index += streak
			streak = 1

	aux_filename = audio_filename.replace("_edited", "_aux")
	final_audio.export(aux_filename, format="wav")

	start = ranges[random_range][0]
	ranges = detect_silence(final_audio, 5, -69.0)	
	
	audio_edited = AudioSegment.from_wav(audio_filename)
	
	audio_start:AudioSegment = audio_edited[:start]
	audio_end:AudioSegment = audio_edited[(start+40+final_audio.duration_seconds*1000):]
	audio_final = audio_start + AudioSegment.silent(duration=20) + final_audio + AudioSegment.silent(duration=20) + audio_end
	out_filename = audio_filename.replace("_edited", "_stego")
	audio_final.export(out_filename, format="wav")
	
	return out_filename

def create_video(image_filename, audio_filename, video_filename):	
	while(check_extension(video_filename, 'video') == False):
		print("ERROR: The filename is not correct. The video must be .avi!")
		video_filename = input("Introduce the filename of the output video: ")

	original_image = image_filename.replace("_stego", "_bgr")

	audio_clip = AudioFileClip(audio_filename)
	clip1 =  ImageClip(original_image).set_duration((audio_clip.duration + 1) // 10)
	clip2 =  ImageClip(image_filename).set_duration(1)
	clip3 =  ImageClip(original_image).set_duration(audio_clip.duration - ((audio_clip.duration + 1) // 10) - 1)
	clips = [clip1, clip2, clip3]
	video_clip = concatenate_videoclips(clips, method='compose')
	video_clip = video_clip.set_audio(audio_clip)
	video_clip.duration = audio_clip.duration
	video_clip.fps = 1
	video_clip.write_videofile(video_filename, codec="rawvideo", audio_codec="pcm_s16le", audio_fps=48000, audio_bitrate="768k", ffmpeg_params=["-ac", "1"])

	print("INFO: The video with the hidden message has been created in: " + video_filename)

#############################
#           DECODE          #
#############################

def get_frame(video_filename, audio_duration):
	try:
		while(check_extension(video_filename, 'video') == False):
			video_filename = input("Introduce the filename of the output video (must be .avi): ")
		cam = cv2.VideoCapture(video_filename)
	except:
		print("ERROR: File not found (" + video_filename + ")!")
		rmtree("./temp")
		exit(1)
	
	frame_to_read = int((audio_duration + 1) // 10 + 1)

	for i in range(frame_to_read):
		ret,frame = cam.read()

	if ret:
		frame_filename = "./temp/frame.png"

		cv2.imwrite(frame_filename, frame)
	  
	cam.release()
	cv2.destroyAllWindows()

	return frame_filename

def get_audio(video_filename):
	try:
		while(check_extension(video_filename, 'video') == False):
			video_filename = input("Introduce the filename of the output video (must be .avi): ")
		video = VideoFileClip(video_filename)
	except:
		print("ERROR: File not found getting audio (" + video_filename + ")!")
		rmtree("./temp")
		exit(1)
	audio = video.audio
	
	audio_filename = './temp/audio.wav'
	audio.write_audiofile(audio_filename, codec="pcm_s16le", fps=48000, bitrate="768k", ffmpeg_params=["-ac", "1"])

	return audio_filename, audio.duration

def decode_image(file, initial_pixel_num, plaintext_password):
	image = cv2.imread(file)

	while(initial_pixel_num == -1):
		initial_pixel_num = int(input('Introduce the number of the initial pixel: '))
	_, current_width, current_height = get_img_coords(image, initial_pixel_num,0)

	message = ''
	final_string = chr(145) * 3

	while(plaintext_password == ''):
		plaintext_password = input('Introduce the password to decode: ')

	password = hashlib.sha256(plaintext_password.encode("utf-8")).digest()

	index = 0
	current_pixel = initial_pixel_num
	while(message[-3:] != final_string):
		pixel = image[current_height][current_width]
		message += chr(pixel[2])
		jump = pixel[0] if pixel[0] != 0 else 100
		current_pixel, current_width, current_height = get_img_coords(image, current_pixel, jump)

	message = message[:-3]
	
	block_size_b64 = 24
	nonce = base64.b64decode(message[:block_size_b64])
	tag = base64.b64decode(message[block_size_b64:block_size_b64 * 2])
	ciphertext = base64.b64decode(message[block_size_b64 * 2:])
	cipher = AES.new(password, AES.MODE_EAX, nonce=nonce)
	try:
		plaintext = cipher.decrypt_and_verify(ciphertext, tag).decode("utf-8")
	except:
		print('ERROR: The password is NOT correct!')
		rmtree("./temp")
		exit(1)

	print('The message is: ' + plaintext)

def decode_audio(audio_filename):
	audio = AudioSegment.from_wav(audio_filename)

	find_silence_range = detect_silence(audio, 18, silence_thresh=-1000000)
	if(find_silence_range[0][0] == 0):
		find_silence_range.pop(0)

	length = find_silence_range[-1][0] - find_silence_range[0][1] 
	if(find_silence_range[-1][1] - find_silence_range[-1][0] > 20):
		length += (find_silence_range[-1][1] - find_silence_range[-1][0] - 20)

	start_point = find_silence_range[0][1]
	
	absolute_silence_range = detect_silence(audio[start_point:start_point+length], 3, silence_thresh=-70)
	final_number = ''

	for i in range(0, len(absolute_silence_range)):
		i_range = absolute_silence_range[i]
		if(i==0 and absolute_silence_range[i][0] >> 0):
			n = i_range[0] / 5
			final_number += int(round(n))*'1'

		if(i >> 0):
			sum = (i_range[0] -  absolute_silence_range[i-1][1])  / 5
			final_number += int(round(sum))*'1'

		n = (i_range[1] - i_range[0]) / 5
		final_number += int(round(n))*'0'

		if(i == len(absolute_silence_range)-1):
			n = (length - i_range[1]) / 5
			final_number += int(round(n))*'1'
	
	initial_pixel = int(final_number, 2)
	if(initial_pixel > 1000000000000):
		raise Exception
	return initial_pixel

def main():
	parser = ArgumentParser(description="Tool to hide a message into a video. More info in README.md", formatter_class=RawTextHelpFormatter)
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument("-e", "--encode", help="Encode a message on a video", action='store_true')
	group.add_argument("-d", "--decode", help="Decode a message from a video", action='store_true')
	parser.add_argument("-v", "--video", help="Video filename (either to decrypt or the desired output of the encription). Must be .avi", type=str)
	parser.add_argument("-i", "--image", help="Image that will contain the message (must be .png)", type=str)
	parser.add_argument("-a", "--audio", help="Audio to hide the starting point (must be .wav)", type=str)
	parser.add_argument("-p", "--password", help="Password to encode or decode", type=str)
	parser.add_argument("-m", "--message", help="Input the message that will be encoded", type=str)
	args = parser.parse_args()

	try:
		if(args.encode):
			os.makedirs("./temp",exist_ok=True)
			password = args.password if args.password else ''
			input_image = args.image if args.image else ''
			message = args.message if args.message else ''
			initial_pixel, encoded_image_filename = encode_image(input_image, password, message)
			
			input_audio = args.audio if args.audio else ''
			encoded_audio_filename = encode_audio(input_audio, initial_pixel)

			video_filename = args.video if args.video else ''
			create_video(encoded_image_filename, encoded_audio_filename, video_filename)
			rmtree("./temp")
		elif(args.decode):
			os.makedirs("./temp",exist_ok=True)
			password = args.password if args.password else ''
			video_filename = args.video if args.video else ''
			audio_filename, audio_duration = get_audio(video_filename)
			frame_filename = get_frame(args.video, audio_duration)
			try:
				initial_pixel = decode_audio(audio_filename)
			except:
				print("ERROR: Something went wrong decoding audio")
				rmtree("./temp")
			# get the message from the image
			decode_image(frame_filename, initial_pixel, password)
			rmtree("./temp")
	except Exception as e:
		print("ERROR: Something went wrong!")
		traceback.print_exc()
		rmtree("./temp")

if __name__ == "__main__":
	main()