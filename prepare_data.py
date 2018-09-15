import json
from os import listdir
from os.path import join, isdir, exists, isfile
from character_table import eprint
import pickle
import re


class SawarefData(object):
    """Given a the path:
    + Extract and merge the JSON files
    """

    def __init__(self, mypath, embeddings_dim,
                 feat_x=["MXpos", "STpos", "AMpos",
                         "FApos", "STaspect", "AMaspect"],
                 strings_x=[],
                 strings_y=[],
                 align_type="",
                 feat_y=["QApos"]):
        # Preparing the big training JSON object
        self.readFiles(mypath, align_type)
        self.embeddings_dim = embeddings_dim
        print(len(self.quran_sent))

        # extract features
        self.features_map_x = feat_x
        self.features_map_y = feat_y

        self.strings_x = strings_x
        self.strings_y = strings_y

        self.features_set_x = dict()
        for f in self.features_map_x:
            self.features_set_x[f] = set()

        self.features_set_y = dict()
        for f in self.features_map_y:
            self.features_set_y[f] = set()

        print('Vectorization...')

        themaxist = max(len(ayah) for ayah in self.quran_sent)
            
        for ayah in self.quran_sent:
            for w in ayah:
                for f in list(w):
                    if f[:2] == "HD":
                        w[f.replace("HD", "QA")] = w[f]
        
        def returnWordLike(prev, default="0"):
            ww = {i: default for i in prev}
            ww["sid"] = prev["sid"]
            ww["aid"] = prev["aid"]
            ww["wid"] = prev["wid"] + 1
            ww["mid"] = 1
            ww["index"] = "-".join(str(x) for x in [ww["sid"], ww["aid"], ww["wid"], ww["mid"]])
            return ww
        # normalize data
        for ayah in self.quran_sent:
            for w in ayah:
                sid, aid, wid, mid = w["0-index"].split("-")
                w["sid"] = sid
                w["aid"] = int(re.sub('[^0-9]', '', aid))
                w["wid"] = int(wid)
                w["mid"] = int(mid)
                for f in self.features_map_y:
                    w[f] = (0 if (w.get(f, 0) == "-----" or
                                  w.get(f, 0) == "-")
                            else w.get(f, 0))
                for f in self.features_map_x:
                    w[f] = (0 if (w.get(f, 0) == "-----" or
                                  w.get(f, 0) == "-")
                            else w.get(f, 0))
            # adding one word to mark end of sentence
            ww = returnWordLike(w, default="0")
            for x in strings_x + strings_y:
                ww[x] = "$"
            ayah.append(ww)

        # cut into 20 each and pad
        newQuran_sent = []
        themax = 20
        for ayah in self.quran_sent:

            for w in ayah:
                w["aid"] = int(str(w["aid"]).replace("P",""))
                w["aid"] *= 100

            noOfWords = len(set(map(
                lambda x: "-".join(x["0-index"].split("-")[:3]),
                ayah)))
            
            if noOfWords < themax:
                for x in range(themax - noOfWords):
                    ww = returnWordLike(ayah[-1], default=-1)
                    for x in strings_x + strings_y:
                        ww[x] = ""
                    ayah.append(ww)
                newQuran_sent.append(ayah)
            else:
                # print("ayah is larger than 20, len=", noOfWords, " ayah=", ayah[0]["0-index"])
                newAyah = []
                for w in ayah:
                    # print(w["wid"] ,"==>", (w["wid"] - 1) % themax + 1, w["mid"], (w["wid"] - 1) % themax == 0 and w["mid"] == 1)
                    # print(w["aid"] ,"==>", w["aid"] + (w["wid"] - 1) // themax)
                    if (w["wid"] - 1) % themax == 0 and w["mid"] == 1:
                        if len(newAyah) != 0:
                            newQuran_sent.append(newAyah)
                        newAyah = []
                    w["aid"] = w["aid"] + (w["wid"] - 1) // themax
                    w["wid"] = ((w["wid"] - 1) % themax) + 1
                    newAyah.append(w)
                for x in range(themax - noOfWords % themax):
                    ww = returnWordLike(newAyah[-1], default=-1)
                    for x in strings_x + strings_y:
                        ww[x] = ""
                    newAyah.append(ww)
                newQuran_sent.append(newAyah)


        del(self.quran_sent)
        self.quran_sent = newQuran_sent

        # build feature sets (uniqe)
        for ayah in self.quran_sent:
            for w in ayah:
                for f in self.features_map_y:
                    if w[f] != 0:
                        self.features_set_y[f].add(w[f])
                for f in self.features_map_x:
                    if w[f] != 0:
                        self.features_set_x[f].add(w[f])

    def get1DJoinedFeatures(self):
        """
            return q&e as a single vector of all morphemes.
            requires aligned morphemes to work properly.
        """
        questions = []
        expected = []
        for ayah in self.quran_sent:
            for w in ayah:
                q = set()
                for f in self.features_map_y:
                    if w[f] != 0:
                        q.add(w[f])
                if len(q) == 0:
                    continue
                expected.append("+".join(q))
                q.clear()
                for f in self.features_map_x:
                    if w[f] != 0:
                        q.add(w[f])
                questions.append("+".join(q))
                q.clear()
        return questions, expected, self.getEmbeddingsFromJson()

    def removeAlignment(self, arr, tallest_seq):
        """
        TODO: complete this function
        """
        new_questions = []
        for quest in arr:
            values = dict()
            for morph in quest:
                for value in morph.split("+"):
                    for feat in self.features_map_x:
                        v = value.replace(feat, "", 1)
                        if v != value:
                            # print(feat, value, v)
                            if values.get(feat) is None:
                                values[feat] = []
                            if v != "_na":
                                values[feat].append(v)
            # print(values)
            newQuest = []
            for x in range(tallest_seq):
                newQuest.append("+".join(([feat + values[feat][x]
                                           for feat in self.features_map_x
                                           if len(values[feat]) > x])))
            new_questions.append(newQuest)
        return new_questions

    def get2DMorphemeJoinedFeatures(self, reverse=True, skipNAs=True):
        """
            return q&e as a 2d vector of words with morphemes as timesteps.
            DOES NOT require aligned morphemes to work properly.
            HOWEVER it DOES NOT remove alignment IF it is already aligned
            SEE removeAlignment()
        """
        questions = []
        expected = []
        position = ""
        word_morphemes_questions = []
        word_morphemes_expected = []
        tallest_seq = 0
        for ayah in self.quran_sent:
            for w in ayah:
                sorah, ayah, wid, mid = w["0-index"].split("-")
                if position != sorah + "-" + ayah + "-" + wid:
                    if reverse:
                        expected.append(word_morphemes_expected[::-1])
                        questions.append(word_morphemes_questions[::-1])
                    else:
                        expected.append(word_morphemes_expected)
                        questions.append(word_morphemes_questions)
                    if len(word_morphemes_expected) > tallest_seq:
                        tallest_seq = len(word_morphemes_expected)
                    if len(word_morphemes_questions) > tallest_seq:
                        tallest_seq = len(word_morphemes_questions)
                    word_morphemes_questions = []
                    word_morphemes_expected = []
                position = sorah + "-" + ayah + "-" + wid
                q = []
                for f in self.features_map_y:
                    if w[f] != 0:
                        q.append(w[f])
                if len(q) == 0:
                    continue
                word_morphemes_expected.append("+".join(q))
                q.clear()
                for f in self.features_map_x:
                    if skipNAs and w[f] != 0:
                        continue
                    q.append(w[f])
                word_morphemes_questions.append("+".join(q))
                q.clear()
        self.pad(questions, tallest_seq)
        self.pad(expected, tallest_seq)
        return (questions[1:], expected[1:], self.getEmbeddingsFromJson(),
                tallest_seq)

    def get2DSentenceJoinedFeatures(self, reverse=True, skipNAs=True):
        """
            return q&e as a 2d vector of words with words
              of one sentence as timesteps.
            REQUIRES aligned morphemes to work properly.
        """
        questions = []
        expected = []
        tallest_seq = 0
        for ayah in self.quran_sent:
            sentence_word_questions = []
            sentence_word_expected = []
            for w in ayah:
                q = []
                for f in self.features_map_y:
                    if w[f] != 0:
                        q.append(w[f])
                if len(q) == 0:
                    continue
                sentence_word_expected.append("+".join(q))
                q.clear()
                for f in self.features_map_x:
                    if skipNAs and w[f] != 0:
                        continue
                    q.append(w[f])
                sentence_word_questions.append("+".join(q))
                q.clear()
            if reverse:
                expected.append(sentence_word_expected[::-1])
                questions.append(sentence_word_questions[::-1])
            else:
                expected.append(sentence_word_expected)
                questions.append(sentence_word_questions)
            if len(sentence_word_expected) > tallest_seq:
                tallest_seq = len(sentence_word_expected)
            if len(sentence_word_questions) > tallest_seq:
                tallest_seq = len(sentence_word_questions)
        self.pad(questions, tallest_seq)
        self.pad(expected, tallest_seq)
        return (questions[1:], expected[1:], self.getEmbeddingsFromJson(),
                tallest_seq)

    def pad(self, arr, tallest_seq):
        # do padding
        for x in arr:
            if len(x) < tallest_seq:
                for i in range(tallest_seq - len(x)):
                    x.append("-")
            if len(x) != tallest_seq:
                eprint("sequence is not padded correctly")
                eprint("tallest_seq=", tallest_seq, "x=", x)
                exit()

    def getEmbeddingsFromJson(self):
        """
            return json
        """
        embeddings = []
        for ayah in self.quran_sent:
            # ayah_emb, words_x, words_y = [], [], []
            for w in ayah:
                if len(w["embeddings"]) < self.embeddings_dim:
                    eprint("embeddings_dim dimension not"
                           "equal the provided ones",
                           len(w["embeddings"]), self.embeddings_dim)
                    exit()
                elif len(w["embeddings"]) > self.embeddings_dim:
                    w["embeddings"] = w["embeddings"][:self.embeddings_dim]

                embeddings.append(w["embeddings"])
        return embeddings

    def readFiles(self, mypath, align_type=""):
        if self.readPickle(mypath, align_type):
            return
        print("Reading from files")
        quran = [f for f in listdir(mypath)
                 if isdir(join(mypath, f)) if (f[:1] == "q" or f[:6] == "fourty") and f[:3] != "q29" and f[:3] != "qur" and f != "fourtyB-P1035"]
        print("Files that do not have ALIGNED", [f for f in quran if not exists(join(mypath, f, align_type + "ALIGNED.json"))])
        quran = [f for f in quran if exists(join(mypath, f, align_type + "ALIGNED.json"))]
        self.quran_sent = []
        for j in quran:
            try:
                js = json.load(open(join(mypath, j,
                                         align_type + "ALIGNED.json")))
                self.quran_sent.append(js)
            except Exception as e:
                print(e)
                print(j)
                raise e
        self.savePickle(mypath, align_type)
        # print(self.quran_sent)

    def getPickleName(self, mypath, align_type="", suffix=".pickle"):
        return mypath.replace("/", ".").strip(".") + "." + align_type + suffix

    def readPickle(self, mypath, align_type=""):
        if isfile(self.getPickleName(mypath, align_type)):
            print("Reading from file:",self.getPickleName(mypath, align_type))
            self.quran_sent = pickle.load(
                open(self.getPickleName(mypath, align_type), mode="rb"),
                encoding="UTF8")
            return True
        return False

    def savePickle(self, mypath, align_type=""):
        pickle.dump(self.quran_sent, open(
            self.getPickleName(mypath, align_type), mode="wb"))

