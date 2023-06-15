
def ffhq():

    list_of_attributes = ['a Bald face', 'a face with Bangs', 'a face with Blond Hair',  'a face with Bushy Eyebrows',  'a face with Beard', 'a face with Pale Skin', 'a Smiling face', 'a face with Lipstick']

    # Not all attributes achieves optimal edits at default beta, you can override the default beta in here.
    attribute_beta = {'a face with Bangs': 0.12, "a face with Pale Skin": 0.12}
    return list_of_attributes, attribute_beta

def afhqcat():

    list_of_attributes = ["a cat with pointed ears", "a cat with open mouth", "a cat with pink nose", "a cat with vertical pupils", "a cat with black fur", "a cat with green eyes", "a cute cat", 'a cat with round face'] #, "a friendly cat", "a cute cat", "a juvenile cat", "a cat with open eyes", 'a cat with round face', 'a furry cat']

    attribute_beta = { "a friendly cat": 0.16, "a juvenile dog":0.16, "a cute dog":0.16,"a cat with long beard":0.12, "a fat cat":0.17}
    
    return list_of_attributes, attribute_beta

def afhqdog():
    
    list_of_attributes = ["a dog with pointed ears", "a dog with open mouth", "a dog with pink nose", "a dog with vertical pupils", "a dog with black fur",  "a cute dog", "a dog with open eyes", 'a dog with round face']
    
    attribute_beta = {"a juvenile dog":0.14, "a cute dog":0.14, "a dog with pointed ears" : 0.14, "a fat dog":0.15,  "a dog with pink nose": 0.1 }

    return list_of_attributes, attribute_beta

def church():

    list_of_attributes = ["a big church", "a white church", "a church with bell", "a church with tower", "a green church" , "a tall church" , "a shabby church", "a church with grass" ]

    attribute_beta = {}

    return list_of_attributes, attribute_beta


def car():

    list_of_attributes = ["a big car", "a white car", " a sports car", "a minivan"]

    attribute_beta = {}

    return list_of_attributes, attribute_beta
