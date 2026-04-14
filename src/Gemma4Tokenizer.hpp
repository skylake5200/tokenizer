#pragma once
#include <sstream>
#include <algorithm>
#include <cstring>

#include "BaseMixinTokenizer.hpp"
#include "utils/object_register.hpp"
#include "utils/sample_log.h"

template <ContentType... Types>
class Gemma4Tokenizer : public BaseMixinTokenizer<Types...>
{
private:
    // Matches google/gemma-4-31B-it chat_template.jinja behavior:
    // remove any "<|channel>...<channel|>" blocks from model turns.
    static std::string strip_thinking(std::string text)
    {
        constexpr const char *kChannelStart = "<|channel>";
        constexpr const char *kChannelEnd = "<channel|>";

        std::string out;
        size_t pos = 0;
        while (true)
        {
            const size_t end = text.find(kChannelEnd, pos);
            const std::string part = (end == std::string::npos) ? text.substr(pos) : text.substr(pos, end - pos);

            const size_t channel_pos = part.find(kChannelStart);
            if (channel_pos != std::string::npos)
            {
                out.append(part.substr(0, channel_pos));
            }
            else
            {
                out.append(part);
            }

            if (end == std::string::npos)
            {
                break;
            }
            pos = end + std::strlen(kChannelEnd);
        }

        BaseMixinTokenizer<Types...>::trim_inplace(out);
        return out;
    }

    static const char *role_to_turn(RoleType role)
    {
        switch (role)
        {
        case SYSTEM:
            return "system";
        case USER:
            return "user";
        case ASSISTANT:
            // Gemma-4 uses "model" instead of "assistant".
            return "model";
        default:
            return "user";
        }
    }

    static const char *type_to_media_token(ContentType type)
    {
        switch (type)
        {
        case IMAGE:
            return "<|image|>";
        case VIDEO:
            return "<|video|>";
        case AUDIO:
            return "<|audio|>";
        default:
            return "";
        }
    }

    static int clamp_media_count(int n)
    {
        // Treat missing/invalid counts as 1 so callers don't have to special-case.
        return (n > 0) ? n : 1;
    }

public:
    std::string apply_chat_template(const std::vector<Content> &contents, bool add_generation_prompt) override
    {
        // check contents type
        for (const auto &content : contents)
        {
            if (!this->support(content.type))
            {
                ALOGE("unsupport content type: %d", content.type);
                return {};
            }
        }

        std::stringstream text;
        text << "<bos>";

        size_t start_idx = 0;
        if (!contents.empty() && contents[0].role == SYSTEM)
        {
            text << "<|turn>system\n";
            std::string system_text = contents[0].data;
            this->trim_inplace(system_text);
            text << system_text;
            text << "<turn|>\n";
            start_idx = 1;
        }

        for (size_t i = start_idx; i < contents.size(); i++)
        {
            const auto &content = contents[i];
            const char *turn_role = role_to_turn(content.role);

            text << "<|turn>" << turn_role << "\n";

            if (content.type == TEXT)
            {
                if (content.role == ASSISTANT)
                {
                    text << strip_thinking(content.data);
                }
                else
                {
                    std::string s = content.data;
                    this->trim_inplace(s);
                    text << s;
                }
            }
            else
            {
                const char *media_token = type_to_media_token(content.type);
                const int n = clamp_media_count(content.num_media);
                for (int j = 0; j < n; j++)
                {
                    text << media_token;
                }

                if (!content.data.empty())
                {
                    if (content.role == ASSISTANT)
                    {
                        text << strip_thinking(content.data);
                    }
                    else
                    {
                        std::string s = content.data;
                        this->trim_inplace(s);
                        text << s;
                    }
                }
            }

            text << "<turn|>\n";
        }

        if (!contents.empty() && contents.back().role == USER && add_generation_prompt)
        {
            // Default: disable thinking by closing the thought channel immediately.
            text << "<|turn>model\n";
            text << "<|channel>thought\n<channel|>";
        }

        return text.str();
    }
};

using gemma4_tokenizer = Gemma4Tokenizer<TEXT, IMAGE, VIDEO, AUDIO>;
REGISTER(Gemma4, gemma4_tokenizer)

