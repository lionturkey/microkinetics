import imageio  

# Build GIF
png_list = [f'runs/{i}.png' for i in range(201)]
with imageio.get_writer('bad_pid.gif', mode='I') as writer:
    for filename in png_list:
        image = imageio.imread(filename)
        writer.append_data(image)
