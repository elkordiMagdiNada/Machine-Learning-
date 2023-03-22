# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import string
import math

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    filenames = ["0.txt", "1.txt", "2.txt", "3.txt", "4.txt", "5.txt", "6.txt", "7.txt", "8.txt","9.txt"]
    languages = ["e", "j", "s"]
    counts_e = [0] * 27
    counts_j = [0] * 27
    counts_s = [0] * 27
    total_e =0;
    total_j = 0;
    total_s = 0;

    for filename in filenames :
        try:
            for language in languages:
                with open("languageID/"+language+filename, 'r') as f:
                    content = f.read()

                    for char in content:
                        if language == 'e' :
                            if char == ' ':
                                counts_e[26] += 1
                                total_e +=1
                            if char.lower() in string.ascii_lowercase:
                                # Increment the count for the corresponding alphabet character
                                index = ord(char.lower()) - ord('a')
                                counts_e[index] += 1
                                total_e += 1
                        if language == 'j' :
                            if char == ' ':
                                counts_j[26] += 1
                                total_j +=1
                            if char.lower() in string.ascii_lowercase:
                                # Increment the count for the corresponding alphabet character
                                index = ord(char.lower()) - ord('a')
                                counts_j[index] += 1
                                total_j += 1
                        if language == 's' :
                            if char == ' ':
                                counts_s[26] += 1
                                total_s +=1
                            if char.lower() in string.ascii_lowercase:
                                # Increment the count for the corresponding alphabet character
                                index = ord(char.lower()) - ord('a')
                                counts_s[index] += 1
                                total_s += 1
        except FileNotFoundError:
            print(f"Error: {filename} not found.")
    counts_e = [(count + 0.5) / (total_e + 27*.5) for count in counts_e]
    counts_j = [(count + 0.5) / (total_j + 27 * .5) for count in counts_j]
    counts_s = [(count + 0.5) / (total_s + 27 * .5) for count in counts_s]
    print (counts_e)
    print (total_e)
    print(counts_j)
    print(total_j)
    print(counts_s)
    print(total_s)

    filenames = ["10.txt", "11.txt", "12.txt", "13.txt", "14.txt", "15.txt", "16.txt", "17.txt", "18.txt", "19.txt"]

    total = 0;
    count_test = [0] * 27
    for language in  languages:
        for filename in filenames:
            try:

                print("languageID/" + language + filename)
                with open("languageID/" + language + filename, 'r') as f:
                    content = f.read()
                    count_test = [0] * 27
                    for char in content:
                        if char == ' ':
                            count_test[26] += 1
                            total += 1
                        if char.lower() in string.ascii_lowercase:
                            # Increment the count for the corresponding alphabet character
                            index = ord(char.lower()) - ord('a')
                            count_test[index] += 1
                            total += 1
                    sum_e = 0
                    sum_j = 0
                    sum_s = 0
                    for i in range(len(count_test)):
                        sum_e += (count_test[i] * math.log(counts_e[i]))
                        sum_j += (count_test[i] * math.log(counts_j[i]))
                        sum_s += (count_test[i] * math.log(counts_s[i]))
                    print(sum_e)
                    # print( math.exp(sum_e))
                    print(sum_j)
                    # print(math.exp(sum_j))
                    print(sum_s)
                    # print(math.exp(sum_s))
                    if (sum_e >= sum_j  and sum_e >= sum_s):
                        print('actual '+language+'Predicted: e')
                    if (sum_j >= sum_e  and sum_j >= sum_s):
                        print('actual '+language+'Predicted: j')
                    if (sum_s >= sum_e  and sum_s >= sum_j):
                        print('actual '+language+'Predicted: s')

                    print(count_test)
            except FileNotFoundError:
                print(f"Error: {filename} not found.")
    # counts = [(count + 0.5) / (total + 27*.5) for count in counts]

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
