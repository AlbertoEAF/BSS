#! /usr/bin/python3

import os.path # To define the special variables

def error(msg):
    print(msg)
    exit(1)

class ConfigParser:
    def __init__(self, configpath, var_parent_localpath = ""):

        self.d = _parse_file_to_dict(configpath)

        # Add Special Variables:
        
        relpath = configpath
        abspath = os.path.abspath(configpath)
        config_name = relpath.split('/')[-1]
        config_extension = ""
        if '.' in config_name:
            config_extension = config_name.split('.')[-1]
            config_name = config_name[:config_name.rfind('.')]
            
        folder_name = "" if '/' not in abspath else abspath.split('/')[-2]

        self.d[':config_relpath'] = relpath
        self.d[':config_abspath'] = abspath
        self.d[':folder_name'] = folder_name
        self.d[':config_name'] = config_name
        self.d[':config_extension'] = config_extension

   
        # Perform recursive replacements and rescans the dictionary over and over until substitutions are exhausted. It checks for loops so it is recursively safe.
        # Doesn't force the search because some parameters might be set on the parent, however all the replacements that can be done now should be done now as they have priority over the parent.
        while dict_replace_bash_vars_once(self.d, force=0):
            continue
            

        # Read the dictionary from the parent and override settings not set in the child.
        if var_parent_localpath and (var_parent_localpath in self.d.keys()):
            parent_localpath = self.d[var_parent_localpath]
            child_folder = configpath[:configpath.rfind("/")]
            parent_path = (child_folder+"/" + parent_localpath).replace("//","/")
            parent_d = _parse_file_to_dict(parent_path)
               
            for parent_key in parent_d:
                if parent_key not in self.d:
                    self.d[parent_key] = parent_d[parent_key]
        
        # Now that the parent has been included we should force the variable substitution.
        dict_replace_bash_vars_once(self.d, force=1)



    def __getitem__(self, key):
        return self.d[key]

    def i(self, key):
        return int(self.d[key])

    def f(self, key):
        return float(self.d[key])

def _parse_file_to_dict(configpath):
    config = open(configpath, 'r')
    lines = [ line.strip() for line in config.readlines() if line.strip() ]
    config.close()

    d = {}

    for line in lines:
        l = line.split('=')
        if len(l) == 2:
            d[l[0].strip()] = l[1].strip()
        elif l[0].strip()[0] != "#":
            error("Invalid line on .cfg file: <{}>".format(line))

    return d


def string_replace_bash_vars(key, string, replace_dictionary, force=1):
    """ Takes a string and replaces the first occurence of a var of the type ${var} in the string by the value of replace_dictionary[var]. """

    MAX_TRY_COUNT = 100 # We should never have to replace this many items in a single string. This is to allow recursive substitution but detect loops.


    old_f = f = 0
    try_count = 0
    replaced = False
    replaced_any = False
    while f > -1 and try_count < MAX_TRY_COUNT:
        try_count += 1
        replaced = False
        old_f = f

        f = string.find("$",old_f)   # Recursive substitution if needed.
        l = string.find("{",f+1)
        r = string.find("}",l+1)
    
        if f < 0:
            break

        if l < 0 or r < 0: 
            print("Invalid string for bash var replacement of format ${var}:",string)
            exit(1)

        var = string[l+1:r]



        # Might as well replace all the occurences already.
        try:
            replacement_string = replace_dictionary[var]

            if string[f:r+1] in replacement_string:
                print("Loop in variable replacements:", key)
                exit(1)
            # Can only replace one pattern at a time because the string might shrink if replacements occur behind the f position.
            string = string[:f] + replacement_string + string[r+1:]
            replaced = True
            replaced_any = True

        except KeyError:
            if force:
                error("Couldn't find the key "+var+" in the replacement dictionary.")
            f += 1 # Couldn't find a thing here, proceed

    if (try_count >= MAX_TRY_COUNT):
        print("Loop detected in vars replacement in string of var:", key)
        exit(1)

    return (string,replaced_any)

def dict_replace_bash_vars_once(dictionary, force=1): 
    """ Takes the dictionary and self replaces the vars known to it running once like a C preprocessor macro system without order. If force is enabled and a substitution can't happen it crashes. Otherwise it goes over it without replacing it (Useful for a recursive engine). """

    replaced_any = True
    while replaced_any:
        replaced_any = False

        for key in dictionary:
            if "$" in dictionary[key]:
                dictionary[key] , replacements_done = string_replace_bash_vars(key,dictionary[key], dictionary, force)
            
                if replacements_done:
                    replaced_any = True

    return replaced_any




        
