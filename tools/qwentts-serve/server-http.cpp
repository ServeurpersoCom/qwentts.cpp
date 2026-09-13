#include "server-http.h"
#include "server.h"

#include "httplib.h"

#include <memory>
#include <future>

class server_http_context::Impl {
public:
    std::unique_ptr<httplib::Server> srv;
};

server_http_context::server_http_context()
    : pimpl(std::make_unique<Impl>())
{}

server_http_context::~server_http_context() = default;

static std::map<std::string, std::string> get_params(const httplib::Request & req) {
    std::map<std::string, std::string> params;
    for (const auto & [key, value] : req.params) {
        params[key] = value;
    }
    for (const auto & [key, value] : req.path_params) {
        params[key] = value;
    }
    return params;
}

static std::map<std::string, std::string> get_headers(const httplib::Request & req) {
    std::map<std::string, std::string> headers;
    for (const auto & [key, value] : req.headers) {
        headers[key] = value;
    }
    return headers;
}

static std::string build_query_string(const httplib::Request & req) {
    std::string qs;
    for (const auto & [key, value] : req.params) {
        if (!qs.empty()) {
            qs += '&';
        }
        qs += httplib::encode_query_component(key) + "=" + httplib::encode_query_component(value);
    }
    return qs;
}

bool server_http_context::init(const server_params & params) {
    port = params.port;
    hostname = params.host;

    auto & srv = pimpl->srv;
    srv.reset(new httplib::Server());

    srv->set_default_headers({{"Server", "qwentts.cpp"}});

    srv->set_read_timeout(60);
    srv->set_write_timeout(120);
    srv->set_payload_max_length(32 * 1024 * 1024);
    srv->set_tcp_nodelay(true);

    srv->set_socket_options([](socket_t sock) {
        int one = 1;
#ifdef _WIN32
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (const char *) &one, sizeof(one));
#else
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
#endif
    });

    srv->set_exception_handler([](const httplib::Request &, httplib::Response & res, const std::exception_ptr & ep) {
        std::string message;
        try {
            std::rethrow_exception(ep);
        } catch (const std::exception & e) {
            message = e.what();
        } catch (...) {
            message = "Unknown Exception";
        }
        res.status = 500;
        res.set_content(message, "text/plain");
    });

    srv->set_error_handler([](const httplib::Request &, httplib::Response & res) {
        if (res.status == 404) {
            res.set_content("{\"error\":{\"message\":\"Not Found\",\"type\":\"not_found_error\",\"code\":404}}",
                            "application/json; charset=utf-8");
        }
    });

    // CORS preflight + middleware
    srv->set_pre_routing_handler([](const httplib::Request & req, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        if (req.method == "OPTIONS") {
            res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
            res.set_content("", "text/html");
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    srv->new_task_queue = [] {
        return new httplib::ThreadPool(4, 128);
    };

    return true;
}

bool server_http_context::start() {
    auto & srv = pimpl->srv;
    if (!srv->bind_to_port(hostname, port)) {
        return false;
    }
    thread = std::thread([this] { pimpl->srv->listen_after_bind(); });
    srv->wait_until_ready();
    listening_address = hostname + ":" + std::to_string(port);
    return true;
}

void server_http_context::stop() const {
    if (pimpl->srv) {
        pimpl->srv->stop();
    }
}

static void set_headers(httplib::Response & res, const std::map<std::string, std::string> & headers) {
    for (const auto & [key, value] : headers) {
        res.set_header(key, value);
    }
}

using server_http_req_ptr = std::unique_ptr<server_http_req>;

static void process_handler_response(server_http_req_ptr && request, server_http_res_ptr & response, httplib::Response & res) {
    if (response->is_stream()) {
        res.status = response->status;
        set_headers(res, response->headers);
        const std::string content_type = response->content_type;
        std::shared_ptr q_ptr = std::move(request);
        std::shared_ptr r_ptr = std::move(response);
        const auto chunked_content_provider = [response = r_ptr](size_t, const httplib::DataSink & sink) -> bool {
            std::string chunk;
            const bool has_next = response->next(chunk);
            if (!chunk.empty()) {
                if (!sink.write(chunk.data(), chunk.size())) {
                    return false;
                }
            }
            if (!has_next) {
                sink.done();
            }
            return has_next;
        };
        const auto on_complete = [request = q_ptr, response = r_ptr](bool) mutable {
            response.reset();
            request.reset();
        };
        res.set_chunked_content_provider(content_type, chunked_content_provider, on_complete);
    } else {
        res.status = response->status;
        set_headers(res, response->headers);
        res.set_content(response->data, response->content_type);
    }
}

void server_http_context::get(const std::string & path, const handler_t & handler) const {
    handlers.emplace(path, handler);
    pimpl->srv->Get(path, [handler](const httplib::Request & req, httplib::Response & res) {
        server_http_req_ptr request = std::make_unique<server_http_req>(server_http_req{
            get_params(req),
            get_headers(req),
            req.path,
            build_query_string(req),
            req.body,
            {},
            req.is_connection_closed
        });
        server_http_res_ptr response = handler(*request);
        process_handler_response(std::move(request), response, res);
    });
}

void server_http_context::post(const std::string & path, const handler_t & handler) const {
    handlers.emplace(path, handler);
    pimpl->srv->Post(path, [handler](const httplib::Request & req, httplib::Response & res) {
        std::string body = req.body;
        std::map<std::string, uploaded_file> files;

        if (req.is_multipart_form_data()) {
            for (const auto & [key, field] : req.form.fields) {
                // form fields appended as newline-separated key=value pairs
                if (!body.empty()) {
                    body += "&";
                }
                body += key + "=" + field.content;
            }
            for (const auto & [key, file] : req.form.files) {
                files[key] = uploaded_file{
                    raw_buffer(file.content.begin(), file.content.end()),
                    file.filename,
                    file.content_type,
                };
            }
        }

        server_http_req_ptr request = std::make_unique<server_http_req>(server_http_req{
            get_params(req),
            get_headers(req),
            req.path,
            build_query_string(req),
            body,
            std::move(files),
            req.is_connection_closed
        });
        server_http_res_ptr response = handler(*request);
        process_handler_response(std::move(request), response, res);
    });
}
