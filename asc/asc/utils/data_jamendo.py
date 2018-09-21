"""
Author: reiniii
"""

import os
import time
import xml.etree.ElementTree as ET
import cPickle as pickle
import numpy as np
from collections import Counter

from audio_processors import RandomAudioSliceProcessor
from data_pools import AugmentedAudioFileMultiClassificationDataPool
from asc.config.settings import DATA_ROOT_JAMENDO

SPEC_BINS = 149
SPEC_CONTEXT = 2 * 313


TAG_DICTIONARY = dict()


def process_tracks(elem):
    tracks = dict()
    tid = None
    tname = None
    for e in elem.getchildren():
        if e.tag == "id":
            tid = int(e.text)
        if e.tag == "name":
            tname = e.text
        tracks[tid] = tname

        tags = []
        if e.tag == "Tags":
            for e_tag in e.getchildren():
                for e_tag2 in e_tag.getchildren():
                    if e_tag2.tag == 'idstr':
                        tags.append(e_tag2.text)

        if tid in TAG_DICTIONARY:
            TAG_DICTIONARY[tid].extend(tags)
        else:
            TAG_DICTIONARY[tid] = tags

    return tracks


def process_albums(elem):
    ntracks = 0
    albums = dict()
    for e in elem.getchildren():
        if e.tag == "album":
            aid = None
            aname = None
            atracks = dict()
            for a in e.getchildren():

                if a.tag == "id":
                    aid = int(a.text)
                if a.tag == "name":
                    aname = a.text
                if a.tag == "Tracks":
                    tracks = []
                    for telem in a.getchildren():
                        tracks.append(process_tracks(telem))

                    for t in tracks:
                        track_id = t.keys()[0]
                        track_name = t.values()[0]
                        atracks[track_id] = track_name

            print "albid: %d,  atracks: %d" % (aid, len(atracks))
            ntracks += len(atracks)
            albums[aid] = (aname, atracks)
    return albums, ntracks


def process_artist(elem):
    ntracks = 0
    albums = dict()
    for e in elem.getchildren():
        if e.tag == "name":
            artist_name = e.text
        if e.tag == "id":
            artist_id = int(e.text)
        if e.tag == "Albums":
            album, ntr = process_albums(e)
            ntracks += ntr
            for album_id, val in album.items():
                album_name, album_tracks = val
                albums[album_id] = (album_name, album_tracks)
    return (artist_id, artist_name, albums, ntracks)


def get_data_by_artist():
    """
    returns dict of
        artist_id: artist_name, artist_albums dicts
    """
    filename = "dbdump_artistalbumtrack.xml"
    num = 450000
    num_tracks = 0
    artists = dict()
    for ev, elem in ET.iterparse(filename):
        if elem.tag == "artist":
            artist_id, artist_name, artist_albums, ntracks = process_artist(elem)
            num_tracks += ntracks
            num -= ntracks
            artists[artist_id] = (artist_name, artist_albums)
        if num <= 0:
            break
    return artists


def trackid_metadata(artists):
    """
    from the artists dict, make a dict by trackids.
    """
    dataset_by_trackid = dict()
    for artist_id, val in artists.iteritems():
        artist_name, albums = val

        for album_id, v in albums.items():
            album_name, tracks = v

            for track_id, track_name in tracks.items():
                metadata = dict()
                metadata['track_name'] = track_name
                metadata['artist_name'] = artist_name
                metadata['artist_id'] = artist_id
                metadata['album_name'] = album_name
                metadata['album_id'] = album_id
                dataset_by_trackid[track_id] = metadata

    return dataset_by_trackid


def run():
    artists = get_data_by_artist()

    tracks = trackid_metadata(artists)
    print len(tracks), len(set(tracks.keys()))

    url_base = "https://storage-new.newjamendo.com/download/track/"
    url_suffix = "/mp32/"
    with open('tracks.p', 'wb') as f:
        pickle.dump(tracks, f, -1)

    urls = [url_base + str(key) + url_suffix for key in tracks.keys()]
    with open('generated_download_urls_450k.txt', 'w') as f:
        for url in urls:
            f.write(url+"\n")

    with open('track_tags.p', 'wb') as f:
        pickle.dump(TAG_DICTIONARY, f, -1)


def load_jamendo_tags(count_thresh=5000, n_seconds=10, n_workers=1):
    """ prepare training set """

    with open(os.path.join(DATA_ROOT_JAMENDO, 'track_tags.p'), 'rb') as f:
        track_tags = pickle.load(f)

    # collect all tags
    all_tags = []
    for tags in track_tags.itervalues():
        all_tags.extend(tags)

    targets = []
    tag_count = 0
    count = Counter(all_tags)
    for k, v in count.iteritems():
        if v > count_thresh:
            tag_count += 1
            targets.append(k)
    targets = np.sort(targets)

    print "Tag Count:", tag_count
    print targets

    # prepare target to network output mapping
    tag_to_id = dict()
    for i, t in enumerate(targets):
        tag_to_id[t] = i

    # collect train songs
    train_songs = []
    train_targets = []
    for song_id, song_tags in track_tags.iteritems():

        if len(np.intersect1d(song_tags, targets)) == 0:
            continue

        song_targets = []
        for st in song_tags:
            if st in targets:
                song_targets.append(tag_to_id[st])

        # build song file path
        song_mp3_path = os.path.join(DATA_ROOT_JAMENDO, "audiofiles", "%d.mp3" % song_id)

        # collect train data
        if os.path.exists(song_mp3_path):
            train_songs.append(song_mp3_path)
            train_targets.append(song_targets)

    # convert to arrays
    train_songs = np.asarray(train_songs, dtype=np.object)
    train_targets = np.asarray(train_targets, dtype=np.object)

    print len(train_songs), "songs collected."

    # init audio processor
    audio_processor = RandomAudioSliceProcessor(n_seconds=n_seconds)

    # init split
    s = int(0.1 * len(train_songs))

    train_pool = AugmentedAudioFileMultiClassificationDataPool(train_songs[:s], train_targets[:s], audio_processor, n_workers, shuffle=True,
                                                               use_cache=False, n_classes=tag_count)
    valid_pool = AugmentedAudioFileMultiClassificationDataPool(train_songs[s:], train_targets[s:], audio_processor, n_workers, shuffle=True,
                                                               use_cache=False, n_classes=tag_count)
    test_pool = AugmentedAudioFileMultiClassificationDataPool(train_songs[s:], train_targets[s:], audio_processor, n_workers, shuffle=True,
                                                              use_cache=False, n_classes=tag_count)

    return {'train': train_pool, 'valid': valid_pool, 'test': test_pool, 'tag_to_id': tag_to_id}


def load_jamendo_artists(count_thresh=250):
    """
    Parameters
    ----------
    count_thresh : int
        Minimum number of songs to consider an artist

    Returns
    -------
    songs_file_paths : list of file paths to song mp3s
    song_meta_data : list of dictionaries (each contains song meta data)
    """

    # load dump file
    with open(os.path.join(DATA_ROOT_JAMENDO, 'tracks.p'), 'rb') as f:
        tracks = pickle.load(f)

    # collect list of all artists
    artists = [v['artist_id'] for v in tracks.itervalues()]

    # count occurrences of artist
    count = Counter(artists)

    # collect artists above threshold
    considered_artists = []
    artist_count = 0
    for artist_id, occ in count.iteritems():
        if occ > count_thresh:
            artist_count += 1
            considered_artists.append(artist_id)

    print "Artist count:", artist_count

    # collect train data
    songs_file_paths = []
    song_meta_data = []
    for song_id, song_data in tracks.iteritems():

        if song_data['artist_id'] not in considered_artists:
            continue

        # build song file path
        song_mp3_path = os.path.join(DATA_ROOT_JAMENDO, "audiofiles", "%d.mp3" % song_id)

        # collect train data
        if os.path.exists(song_mp3_path):
            songs_file_paths.append(song_mp3_path)
            song_meta_data.append(song_data)

    print len(songs_file_paths), "songs collected."

    return songs_file_paths, song_meta_data


if __name__ == "__main__":
    """ main """

    # run()

    # --- investigate tags ---
    data = load_jamendo_tags(count_thresh=3000, n_seconds=20, n_workers=10)

    def prepare(X, y):
        """ Prepare data for network processing """
        shape = X.shape
        X = X.reshape((shape[0], 1, shape[1], shape[2]))

        return X, y

    from lasagne_wrapper.batch_iterators import BatchIterator
    bi = BatchIterator(batch_size=10, prepare=prepare, k_samples=100, shuffle=True)

    for e in xrange(5):
        start = time.time()
        for i, (x, y) in enumerate(bi(data['train'])):
            stop = time.time()
            print e, i, x.shape, y.shape, stop-start, "sec batch time"
            start = stop

    # --- investigate artists ---
    # load_jamendo_artists(count_thresh=450)
    #
    # with open(os.path.join(DATA_ROOT_JAMENDO, 'tracks.p'), 'rb') as f:
    #     tracks = pickle.load(f)
    #
    # artists = [v['artist_id'] for v in tracks.itervalues()]
    #
    # count = Counter(artists)
    #
    # sorted_counts = np.sort(count.values())
    #
    # thresh = 250
    # print np.sum(sorted_counts > thresh), thresh * np.sum(sorted_counts > thresh)
