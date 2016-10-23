import logging
import logging.handlers
from wsgiref.simple_server import make_server
from urlparse import parse_qs
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
import numpy

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
        'y_shape': 33,
        'weights_name': WEIGHTS_DIRECTORY + 'dante-1.4319.hdf5',
        'char_to_int': {'\n': 0, ' ': 1, '"': 2, "'": 3, ',': 4, '.': 5, '?': 6, 'a': 7, 'c': 9, 'b': 8, 'e': 11, 'd': 10, 'g': 13, 'f': 12, 'i': 15, 'h': 14, 'k': 17, 'j': 16, 'm': 19, 'l': 18, 'o': 21, 'n': 20, 'q': 23, 'p': 22, 's': 25, 'r': 24, 'u': 27, 't': 26, 'w': 29, 'v': 28, 'y': 31, 'x': 30, 'z': 32},
        'int_to_char': {0: '\n', 1: ' ', 2: '"', 3: "'", 4: ',', 5: '.', 6: '?', 7: 'a', 8: 'b', 9: 'c', 10: 'd', 11: 'e', 12: 'f', 13: 'g', 14: 'h', 15: 'i', 16: 'j', 17: 'k', 18: 'l', 19: 'm', 20: 'n', 21: 'o', 22: 'p', 23: 'q', 24: 'r', 25: 's', 26: 't', 27: 'u', 28: 'v', 29: 'w', 30: 'x', 31: 'y', 32: 'z'}
    },
    'shakespeare': {
        'y_shape': 32,
        'weights_name': WEIGHTS_DIRECTORY + 'shakespeare-1.3267.hdf5',
        'char_to_int': {'\n': 0, ' ': 1, "'": 2, ',': 3, '.': 4, '?': 5, 'a': 6, 'c': 8, 'b': 7, 'e': 10, 'd': 9, 'g': 12, 'f': 11, 'i': 14, 'h': 13, 'k': 16, 'j': 15, 'm': 18, 'l': 17, 'o': 20, 'n': 19, 'q': 22, 'p': 21, 's': 24, 'r': 23, 'u': 26, 't': 25, 'w': 28, 'v': 27, 'y': 30, 'x': 29, 'z': 31},
        'int_to_char': {0: '\n', 1: ' ', 2: "'", 3: ',', 4: '.', 5: '?', 6: 'a', 7: 'b', 8: 'c', 9: 'd', 10: 'e', 11: 'f', 12: 'g', 13: 'h', 14: 'i', 15: 'j', 16: 'k', 17: 'l', 18: 'm', 19: 'n', 20: 'o', 21: 'p', 22: 'q', 23: 'r', 24: 's', 25: 't', 26: 'u', 27: 'v', 28: 'w', 29: 'x', 30: 'y', 31: 'z'}
    },
    'rowling': {
        'y_shape': 35,
        'weights_name': WEIGHTS_DIRECTORY + 'rowling-1.2660.hdf5',
        'char_to_int': {'\t': 0, '\n': 1, ' ': 2, '"': 3, "'": 4, ',': 5, '.': 6, '?': 7, '\\': 8, 'a': 9, 'c': 11, 'b': 10, 'e': 13, 'd': 12, 'g': 15, 'f': 14, 'i': 17, 'h': 16, 'k': 19, 'j': 18, 'm': 21, 'l': 20, 'o': 23, 'n': 22, 'q': 25, 'p': 24, 's': 27, 'r': 26, 'u': 29, 't': 28, 'w': 31, 'v': 30, 'y': 33, 'x': 32, 'z': 34},
        'int_to_char': {0: '\t', 1: '\n', 2: ' ', 3: '"', 4: "'", 5: ',', 6: '.', 7: '?', 8: '\\', 9: 'a', 10: 'b', 11: 'c', 12: 'd', 13: 'e', 14: 'f', 15: 'g', 16: 'h', 17: 'i', 18: 'j', 19: 'k', 20: 'l', 21: 'm', 22: 'n', 23: 'o', 24: 'p', 25: 'q', 26: 'r', 27: 's', 28: 't', 29: 'u', 30: 'v', 31: 'w', 32: 'x', 33: 'y', 34: 'z'}
    },
    'poe': {
        'y_shape': 34,
        'weights_name': WEIGHTS_DIRECTORY + 'poe-1.0889.hdf5',
        'char_to_int': {'\n': 0, ' ': 1, '"': 2, "'": 3, ',': 4, '.': 5, '?': 6, 'a': 8, '`': 7, 'c': 10, 'b': 9, 'e': 12, 'd': 11, 'g': 14, 'f': 13, 'i': 16, 'h': 15, 'k': 18, 'j': 17, 'm': 20, 'l': 19, 'o': 22, 'n': 21, 'q': 24, 'p': 23, 's': 26, 'r': 25, 'u': 28, 't': 27, 'w': 30, 'v': 29, 'y': 32, 'x': 31, 'z': 33},
        'int_to_char': {0: '\n', 1: ' ', 2: '"', 3: "'", 4: ',', 5: '.', 6: '?', 7: '`', 8: 'a', 9: 'b', 10: 'c', 11: 'd', 12: 'e', 13: 'f', 14: 'g', 15: 'h', 16: 'i', 17: 'j', 18: 'k', 19: 'l', 20: 'm', 21: 'n', 22: 'o', 23: 'p', 24: 'q', 25: 'r', 26: 's', 27: 't', 28: 'u', 29: 'v', 30: 'w', 31: 'x', 32: 'y', 33: 'z'}
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

        text = parsed_body.get('text', [''])[0]
        author = parsed_body.get('author', [''])[0]
        output_length = int(parsed_body.get('length', [''])[0])

        if text.strip() != "":
            response_body = translate(author, text, output_length)
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


def translate(author, text, output_length):
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

    return generate_chars(model, pattern, int_to_char, output_length)


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
