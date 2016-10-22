import logging
import logging.handlers

from wsgiref.simple_server import make_server

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Handler 
#LOG_FILE = '/opt/python/log/sample-app.log'
#handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=1048576, backupCount=5)
#handler.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add Formatter to Handler
#handler.setFormatter(formatter)

# add Handler to Logger
#logger.addHandler(handler)

FILE = 'index.html'


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
            if path.startswith('/translate'):
                request_body_size = int(environ['CONTENT_LENGTH'])
                request_body = environ['wsgi.input'].read(request_body_size).decode()
                logger.info("Received message: %s" % request_body)
            elif path == '/scheduled':
                logger.info("Received task %s scheduled at %s", environ['HTTP_X_AWS_SQSD_TASKNAME'], environ['HTTP_X_AWS_SQSD_SCHEDULED_AT'])
        except (TypeError, ValueError):
            logger.warning('Error retrieving request body for async work.')
        response_body = ''
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


if __name__ == '__main__':
    httpd = make_server('', 8000, application)
    print("Serving on port 8000...")
    httpd.serve_forever()
