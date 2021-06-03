# To use this method, there are standard rules for model defined .py file.
# All layer will be defined in __init__ of the model class
# In forward method, you should use 'layer = self.layer(input)' formulation
# and also, define forward will be positioned at the last of the .py code.

def auto_checkpointing(filename, new_fileName):
    outputTxt = ""
    trigger = -1
    with open(filename, 'r') as fp:
        codeStrings = fp.readlines()
    lineLength = codeStrings.__len__()
    lineCount = -1
    del_idx_dict = dict()

    for i in codeStrings:
        lineCount += 1
        # when we find the code start with 'def forward', trigger becomes 'On' state until find 'return'
        # trigger becomes On (0) when only the code reader reads forward define parts
        if 'def forward' in i:
            trigger = 0
        if (trigger != -1) and 'return' in i:
            trigger = -1

        # when the code is not in forward define parts, no needs to checkpoint
        if trigger == -1:
            outputTxt += i
            continue
        else: #if trigger is On, find checkpoint
            # inside 'def forward'
            #print("lineCount " + str(lineCount) + " : " + codeStrings[lineCount])
            if 'self.' in i:
                if lineCount % 3 != 0:
                    indentLength = len(i) - len(i.lstrip())
                    indent = i[0 : indentLength]
                    outputLayer = i[indentLength : i.find('=')]
                    currentLayer = i[i.find('=')+1 : i.find('(')]
                    inputLayer = i[i.find('(')+1 : i.find(')')]
                    checkpointedLine = indent + outputLayer + "=" + "torch.utils.checkpoint.checkpoint(" + currentLayer + "," + inputLayer + ")\n"
                    outputTxt += checkpointedLine
                    # outputTxt = outputTxt + indent + "torch.cuda.empty_cache()\n"
                    continue
            outputTxt += i
    # only checkpointed, not del & empty_cache
    with open(new_fileName, 'w') as fp2:
        fp2.write(outputTxt)
