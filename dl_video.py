import pytube

url = "https://www.youtube.com/watch?v=kwQ84hVrMJo"

# download the video in 360p 
video = pytube.YouTube(url).streams.get_lowest_resolution()
downloaded_video = video.download()