import os
import json


def simplify(path, newpath):
    filenames = os.listdir(path)

    playlists = []

    for filename in sorted(filenames):
        if filename.startswith("mpd.slice.") and filename.endswith(".json"):
            fullpath = os.sep.join((path, filename))
            with open(fullpath) as f:
                mpd_slice = json.load(f)
                for p in mpd_slice["playlists"]:
                    to_add = {"num_tracks": p["num_tracks"], "pid": p["pid"]}
                    if "name" in p:
                        to_add["name"] = p["name"]
                    to_add["tracks"] = [t["track_uri"][14:] for t in p["tracks"]]
                    playlists.append(to_add)

    with open(newpath, "w") as f:
        json.dump({"playlists": playlists}, f)


if __name__ == "__main__":
    simplify("dataset/spotify/data", "dataset/simplified.json")
