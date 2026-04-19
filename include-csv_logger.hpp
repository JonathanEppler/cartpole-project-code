#pragma once
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>

class CsvLogger {
public:
    explicit CsvLogger(const std::string& path)
        : file_(path)
    {
        if (!file_) throw std::runtime_error("Failed to open CSV file: " + path);
    }

    void write_header(const std::vector<std::string>& cols) {
        for (size_t i = 0; i < cols.size(); ++i) {
            file_ << cols[i] << (i + 1 < cols.size() ? "," : "\n");
        }
    }

    template <typename... Ts>
    void write_row(Ts... vals) {
        write_vals(vals...);
        file_ << "\n";
    }

private:
    std::ofstream file_;

    void write_vals() {}

    template <typename T, typename... Ts>
    void write_vals(T v, Ts... rest) {
        file_ << v;
        if constexpr (sizeof...(rest) > 0) file_ << ",";
        write_vals(rest...);
    }
};
