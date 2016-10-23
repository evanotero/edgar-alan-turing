import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
import logging
import logging.handlers
from urlparse import parse_qs

from wsgiref.simple_server import make_server

# # Create logger
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
#
# # Handler
# LOG_FILE = '/opt/python/log/sample-app.log'
# handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=1048576, backupCount=5)
# handler.setLevel(logging.INFO)
#
# # Formatter
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#
# # Add Formatter to Handler
# handler.setFormatter(formatter)
#
# # add Handler to Logger
# logger.addHandler(handler)

OUTPUT_LENGTH = 200
SEQUENCE_LENGTH = 100
FILE = 'index.html'
BAD_TEXT = '/:;][{}-=+)(*&^%-_$#@!~1234567890'
WEIGHTS_DIRECTORY = "scripts/model_weights/"
author_vars = {
    'tolkien': {
        'y_shape': 42,
        'weights_name': WEIGHTS_DIRECTORY + 'tolkien-0.9459.hdf5',
        'char_to_int': {'\n': 0, '!': 2, ' ': 1, '"': 3, "'": 4, '\xa9': 39, '-': 6, ',': 5, '.': 7, '\xb3': 40, ';': 9, ':': 8, '?': 10, '\xc3': 41, '_': 11, 'a': 13, '`': 12, 'c': 15, 'b': 14, 'e': 17, 'd': 16, 'g': 19, 'f': 18, 'i': 21, 'h': 20, 'k': 23, 'j': 22, 'm': 25, 'l': 24, 'o': 27, 'n': 26, 'q': 29, 'p': 28, 's': 31, 'r': 30, 'u': 33, 't': 32, 'w': 35, 'v': 34, 'y': 37, 'x': 36, 'z': 38},
        'int_to_char': {0: '\n', 1: ' ', 2: '!', 3: '"', 4: "'", 5: ',', 6: '-', 7: '.', 8: ':', 9: ';', 10: '?', 11: '_', 12: '`', 13: 'a', 14: 'b', 15: 'c', 16: 'd', 17: 'e', 18: 'f', 19: 'g', 20: 'h', 21: 'i', 22: 'j', 23: 'k', 24: 'l', 25: 'm', 26: 'n', 27: 'o', 28: 'p', 29: 'q', 30: 'r', 31: 's', 32: 't', 33: 'u', 34: 'v', 35: 'w', 36: 'x', 37: 'y', 38: 'z', 39: '\xa9', 40: '\xb3', 41: '\xc3'}
    },
    'dante': {

    },
    'shakespeare': {

    },
    'rowling': {

    },
    'poe': {

    }
}


def application(environ, start_response):
    path = environ['PATH_INFO'].lstrip('/')  # get path
    method = environ['REQUEST_METHOD']

    if "js" in path:  # handle query for
        status = '200 OK'  # files in /static
        headers = [('Content-type', 'text/javascript')]
        start_response(status, headers)
        f2serv = file(path, 'r')  # read file
        return environ['wsgi.file_wrapper'](f2serv)  # return file

    if "css" in path:  # handle query for
        status = '200 OK'  # files in /static
        headers = [('Content-type', 'text/css')]
        start_response(status, headers)
        f2serv = file(path, 'r')  # read file
        return environ['wsgi.file_wrapper'](f2serv)  # return file

    if "svg" in path:
        status = '200 OK'  # files in /static
        headers = [('Content-type', 'image/svg+xml')]
        start_response(status, headers)
        f2serv = file(path, 'r')  # read file
        return environ['wsgi.file_wrapper'](f2serv)  # return file

    if "img" in path:
        status = '200 OK'  # files in /static
        headers = [('Content-type', 'image/jpg')]
        start_response(status, headers)
        f2serv = file(path, 'r')  # read file
        return environ['wsgi.file_wrapper'](f2serv)  # return file

    if method == 'POST':
        try:
            request_body_size = int(environ['CONTENT_LENGTH'])
            request_body = environ['wsgi.input'].read(request_body_size)
        except (TypeError, ValueError):
            request_body = "0"
        parsed_body = parse_qs(request_body)

        text = parsed_body.get('text', [''])[0]  # Returns the first value
        author = parsed_body.get('author', [''])[0]

        if text.strip() != "":
            response_body = translate(author, text)
        else:
            response_body = "No input text."

        status = '200 OK'
        headers = [('Content-type', 'text/html')]
        start_response(status, headers)
    else:
        response_body = open(FILE).read()  # the html file itself
        status = '200 OK'
        headers = [('Content-type', 'text/html'),
                   ('Content-Length', str(len(response_body)))]
        start_response(status, headers)
    return [response_body]


def translate(author, text):
    vars = author_vars.get(author)
    char_to_int = vars.get("char_to_int")
    int_to_char = vars.get("int_to_char")

    text = text.lower()
    text = text.translate(None, BAD_TEXT)
    if len(text) < 100:
        num_spaces = 100 - len(text)
        for i in range(num_spaces):
            text = " " + text
    else:
        text = text[len(text) - 100:len(text)]

    pattern = [char_to_int[letter] for letter in text]

    # define the LSTM model
    model = Sequential()
    model.add(LSTM(256, input_shape=(100, 1), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(vars.get("y_shape"), activation='softmax'))

    model.load_weights(vars.get("weights_name"))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return generate_chars(model, pattern, int_to_char, OUTPUT_LENGTH)


def generate_ints(text, chars):
    dataX = []
    dataY = []
    for i in range(len(text) - SEQUENCE_LENGTH):
        seq_in = text[i:i + SEQUENCE_LENGTH]
        seq_out = text[i + SEQUENCE_LENGTH]
        dataX.append([chars[char] for char in seq_in])
        dataY.append(chars[seq_out])
    return dataX, dataY


def generate_chars(model, pattern, int_to_char, length):
    output = ""
    for _ in range(length):
        x = numpy.reshape(pattern, (1, len(pattern), 1))
        n_vocab = len(int_to_char)
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        index = numpy.argmax(prediction)
        result = int_to_char[index]
        # seq_in = [int_to_char[value] for value in pattern]
        output += result
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    return output


if __name__ == '__main__':
    httpd = make_server('', 8000, application)
    print("Serving on port 8000...")
    httpd.serve_forever()
