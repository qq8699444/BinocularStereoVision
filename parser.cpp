
#include <fstream>
#include <sstream>
#include "parser.h"

template <class Type>
Type stringToNum(const string& str)
{
    istringstream iss(str);
    Type num;
    iss >> num;
    return num;
}

vector<std::string> stringSplit(const string& str,const char delim)
{
    vector <string> tokens;
    stringstream check1(str);

    string intermediate;

    // Tokenizing w.r.t. space ' '
    while(getline(check1, intermediate, delim))
    {
        tokens.push_back(intermediate);
    }

    return tokens;
}

std::vector<Point2d>    prasePointSFromFile(const string& filename)
{
    std::ifstream     fs(filename);
    std::vector<Point2d> result;

    if (fs.is_open())
    {
        const int LINE_LENGTH = 0x100;
        char str[LINE_LENGTH];
        std::vector<float> corrds;
        while (fs.getline(str,LINE_LENGTH))
        {
            //cout << " Read from file: " << str << endl;
            auto tokens = stringSplit(str);
            for (auto token:tokens)
            {
                if (token.empty())
                {
                    continue;
                }
                corrds.push_back(stringToNum<double>(token.c_str()));
            }
        }

        fs.close();
        if (corrds.size() %2 != 0)
        {
            std::cout << "x y mismatch" << endl;
            return result;
        }

        result.reserve(corrds.size() / 2);
        for (int i = 0;i < corrds.size() / 2;i++)
        {
            result.push_back(Point2f(corrds[2*i],corrds[2*i+1]));
        }
    }
    else
    {
        std::cout << "Error opening file" << endl;
        
    }

    
    return result;
}
