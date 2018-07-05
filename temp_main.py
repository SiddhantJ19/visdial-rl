from visdial.models.answerer import Answerer

from nltk.tokenize import word_tokenize
from six.moves import range
from skimage.transform import resize
from torch.autograd import Variable

import json
import random
import numpy as np
import skimage.io
import time
import torch.nn as nn
import torchvision
import torch


class VisualDialog(object):
    def __init__(self):
        self.params = {
            'inputJson': "data/visdial/chat_processed_params.json",
            'useGPU': False,
            # A-Bot checkpoint
            'startFrom': "./checkpoints/abot_sl_ep60.vd",
            'beamSize': 5,
        }

        manualSeed = 1597
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)

        if self.params['useGPU']:
            torch.cuda.manual_seed_all(manualSeed)

        print('Loading json file: ' + self.params['inputJson'])
        with open(self.params['inputJson'], 'r') as fileId:
            self.info = json.load(fileId)

        wordCount = len(self.info['word2ind'])
        # Add <START> and <END> to vocabulary
        self.info['word2ind']['<START>'] = wordCount + 1
        self.info['word2ind']['<END>'] = wordCount + 2
        startToken = self.info['word2ind']['<START>']
        endToken = self.info['word2ind']['<END>']

        # Padding token is at index 0

        vocabSize = wordCount + 3

        print('Vocab size with <START>, <END>: %d' % vocabSize)

        # Construct the reverse map

        self.info['ind2word'] = {
            int(ind): word
            for word, ind in self.info['word2ind'].items()
        }

        # load aBot

        self.aBot = self.load_model(self.params, 'abot')

        assert self.aBot.encoder.vocabSize == vocabSize, "Vocab size mismatch!"

        self.aBot.eval()

        # load pre-trained VGG 19

        print("Loading image feature extraction model")

        self.feat_extract_model = torchvision.models.vgg19(pretrained=True)

        self.feat_extract_model.classifier = nn.Sequential(
            *list(self.feat_extract_model.classifier.children())[:-3])

        self.feat_extract_model.eval()

        if self.params['useGPU']:

            self.feat_extract_model.cuda()

        print("Done!")

    def load_model(self, params, agent='abot'):

        # should be everything used in encoderParam, decoderParam below

        encoderOptions = [
            'encoder',
            'vocabSize',
            'embedSize',
            'rnnHiddenSize',
            'numLayers',
            'useHistory',
            'useIm',
            'imgEmbedSize',
            'imgFeatureSize',
            'numRounds',
            'dropout',
        ]

        decoderOptions = [
            'decoder',
            'vocabSize',
            'embedSize',
            'rnnHiddenSize',
            'numLayers',
            'dropout',
        ]

        modelOptions = encoderOptions + decoderOptions

        mdict = None

        startArg = 'startFrom' if agent == 'abot' else 'qstartFrom'

        assert self.params[startArg], "Need checkpoint for {}".format(agent)

        if self.params[startArg]:

            print('Loading model (weights and config) from {}'.format(
                self.params[startArg]))

            if self.params['useGPU']:

                mdict = torch.load(self.params[startArg])

            else:

                mdict = torch.load(
                    self.params[startArg],
                    map_location=lambda storage, location: storage)

            # Model options is a union of standard model options defined

            # above and parameters loaded from checkpoint

            modelOptions = list(set(modelOptions).union(set(mdict['params'])))

            for opt in modelOptions:

                if opt not in params:

                    self.params[opt] = mdict['params'][opt]

                elif self.params[opt] != mdict['params'][opt]:

                    # Parameters are not overwritten from checkpoint

                    pass

        # Initialize model class

        encoderParam = {k: self.params[k] for k in encoderOptions}

        decoderParam = {k: self.params[k] for k in decoderOptions}

        encoderParam['startToken'] = encoderParam['vocabSize'] - 2

        encoderParam['endToken'] = encoderParam['vocabSize'] - 1

        decoderParam['startToken'] = decoderParam['vocabSize'] - 2

        decoderParam['endToken'] = decoderParam['vocabSize'] - 1

        encoderParam['type'] = self.params['encoder']

        decoderParam['type'] = self.params['decoder']

        encoderParam['isAnswerer'] = True

        model = Answerer(encoderParam, decoderParam)

        if self.params['useGPU']:

            model.cuda()

        if mdict:

            model.load_state_dict(mdict['model'])

        print("Loaded agent {}".format(agent))

        return model

    def transform(self, img):
        """

        Process image

        """

        img = img.astype("float") / 255

        img = resize(img, (224, 224), mode='constant')

        img[:, :, 0] -= 0.485

        img[:, :, 1] -= 0.456

        img[:, :, 2] -= 0.406

        return img.transpose([2, 0, 1])

    def ind_map(self, words):
        return np.array(  # noqa
            [
                self.info['word2ind'].get(word, self.info['word2ind']['UNK'])
                for word in words
            ],
            dtype='int64')

    def tokenize(self, string):
        return ['<START>'] + word_tokenize(string) + ['<END>']  # noqa

    def var_map(self, tensor):

        if self.params['useGPU']:

            tensor = tensor.cuda()

        return Variable(tensor.unsqueeze(0), volatile=True)

    # Helper functions for converting tensors to words

    def to_str_pred(self, w, l):
        return str(" ".join([
            self.info['ind2word'][x] for x in list(
                filter(  # noqa
                    lambda x: x > 0,
                    w.data.cpu().numpy()))
        ][:l.data.cpu()[0]]))[8:]  # noqa

    def to_str_gt(self, w):
        return str(" ".join([
            self.info['ind2word'][x] for x in filter(  # noqa
                lambda x: x > 0,
                w.data.cpu().numpy())
        ]))[8:-6]  # noqa

    def predict(self, question, img_path, caption, dialog):

        now = time.time()

        beamSize = 5

        # # Process Image Path

        raw_img = self.transform(skimage.io.imread(img_path))

        # Process caption

        caption_tokens = self.tokenize(caption)

        caption = self.ind_map(caption_tokens)

        # Process history

        h_question_tokens = []

        h_questions = []

        h_answer_tokens = []

        h_answers = []

        print("L215: ", time.time() - now)

        for round_idx, item in enumerate(dialog):

            ans_tokens = self.tokenize(item['answer'])

            h_answer_tokens.append(ans_tokens)

            h_answers.append(self.ind_map(ans_tokens))

            ques_tokens = self.tokenize(item['question'])

            h_question_tokens.append(ques_tokens)

            h_questions.append(self.ind_map(ques_tokens))

        print("L225: ", time.time() - now)
        # Process question

        question_tokens = self.tokenize(question)

        question = self.ind_map(question_tokens)

        img_tensor = self.var_map(torch.from_numpy(raw_img).float())

        img_feats = self.feat_extract_model(img_tensor)

        _norm = torch.norm(img_feats, p=2, dim=1)

        img_feats = img_feats.div(_norm.expand_as(img_feats))

        caption_tensor = self.var_map(torch.from_numpy(caption))

        caption_lens = self.var_map(torch.LongTensor([len(caption)]))

        question_tensor = self.var_map(torch.from_numpy(question))
        question_lens = self.var_map(torch.LongTensor([len(question)]))

        print("L242: ", time.time() - now)

        hist_ans_tensors = [
            self.var_map(torch.from_numpy(ans)) for ans in h_answers
        ]

        hist_ans_lens = [
            self.var_map(torch.LongTensor([len(h_ans)]))
            for h_ans in h_answer_tokens
        ]

        hist_ques_tensors = [
            self.var_map(torch.from_numpy(ques)) for ques in h_questions
        ]

        hist_ques_lens = [
            self.var_map(torch.LongTensor([len(h_ques)]))
            for h_ques in h_question_tokens
        ]

        print("L252: ", time.time() - now)

        self.aBot.eval(), self.aBot.reset()

        self.aBot.observe(
            -1,
            image=img_feats,
            caption=caption_tensor,
            captionLens=caption_lens)

        print("L260: ", time.time() - now)

        numRounds = len(dialog)

        beamSize = self.params['beamSize']

        for round in range(numRounds):

            self.aBot.observe(
                round,
                ques=hist_ques_tensors[round],
                quesLens=hist_ques_lens[round])

            self.aBot.observe(
                round,
                ans=hist_ans_tensors[round],
                ansLens=hist_ans_lens[round])

            self.aBot.forward()

            answers, ansLens = self.aBot.forwardDecode(
                inference='greedy', beamSize=beamSize)

        # After processing history
        self.aBot.observe(
            numRounds, ques=question_tensor, quesLens=question_lens)

        answers, ansLens = self.aBot.forwardDecode(
            inference='greedy', beamSize=beamSize)

        print("L286: ", time.time() - now)
        print("Q%d: " % (numRounds + 1), self.to_str_gt(question_tensor[0]))
        print("A%d: " % (numRounds + 1),
              self.to_str_pred(answers[0], ansLens[0]))
        current_dialog = {}
        current_dialog['question'] = self.to_str_gt(question_tensor[0])
        current_dialog['answer'] = self.to_str_pred(answers[0], ansLens[0])
        print(time.time() - now)
        # dialog.append(current_dialog)
        return dialog + [current_dialog]


def main():
    DIALOG = []
    visdial_model = VisualDialog()

    while (True):
        CAPTION = "a cat sitting on top of a refrigerator"
        QUESTION = input("Enter the question: ")
        IMG_PATH = "demo/img.jpg"
        if QUESTION == "EXIT":
            break
        dialog = visdial_model.predict(QUESTION, IMG_PATH, CAPTION, DIALOG)
        DIALOG = dialog
        print(dialog)


if __name__ == "__main__":

    main()
