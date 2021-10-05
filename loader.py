import csv
from scipy.io.wavfile import read


def load_audio_files_from_dir(classifications_path, sample_width_s, overlap_frac = 0):
    keys = []
    data_files = []
    with open("./raw_audio/classifications.txt") as f:
        reader = csv.reader(f, delimiter=',')
        for idx, row in enumerate(reader):
            if idx == 0:
                keys = row
                # print("Opened classifications.txt file, found cols \n" + str(row))
            else:
                # load all columns from csv file
                data_files.append({keys[index].strip(): row[index].strip() for index in range(len(row))})
                # read audio file and sampling rate into dictionary
                data_files[-1]["rate"], data_files[-1]["data"] = read("./raw_audio/" + data_files[-1]["filename"])

    # Chop files into chunks
    data_vectors = []
    classifications = []
    sample_width_s = 0.1

    # pdb.set_trace()
    for f in data_files:
        start_index = int(f["rate"] * float(f["start"]))
        end_index = int(f["rate"] * float(f["end"]))
        points_per_sample = int(sample_width_s * f["rate"])
        samples_in_file = int((end_index - start_index) / points_per_sample)
        for index in range(samples_in_file):
            audio_sample = f["data"][
                           (start_index + index * points_per_sample):(start_index + (index + 1) * points_per_sample)]
            data_vectors.append([audio[0] for audio in audio_sample])  # drop spurious second channel
            classifications.append(f["wear"])
    return data_vectors, classifications