#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace das_terminal {

struct PtyProcessOptions {
    std::string command_line;
    std::vector<std::string> arguments;
    std::string working_directory;
    int columns = 80;
    int rows = 25;
};

enum class PtyReadStatus : uint8_t {
    data,
    idle,
    closed,
    error,
};

class PtyProcess {
public:
    virtual ~PtyProcess() = default;

    virtual uint32_t processId() const = 0;
    virtual PtyReadStatus read(std::string & bytes, std::string & error,
                               size_t maximum_bytes = 64 * 1024) = 0;
    virtual bool write(const uint8_t * bytes, size_t count, std::string & error) = 0;
    bool write(const std::string & bytes, std::string & error) {
        return write(reinterpret_cast<const uint8_t *>(bytes.data()), bytes.size(), error);
    }
    virtual bool resize(int columns, int rows, std::string & error) = 0;
    virtual bool wait(uint32_t timeout_ms, uint32_t & exit_code, std::string & error) = 0;
    virtual bool terminate(uint32_t exit_code, std::string & error) = 0;
};

std::unique_ptr<PtyProcess> launchPtyProcess(
    const PtyProcessOptions & options, std::string & error);

} // namespace das_terminal
