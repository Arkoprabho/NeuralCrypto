"""
Acts as the starting point for the program
The main objective of the program is to use a neural network based encryption algorithm.
"""
import server
import client

if __name__ == '__main__':
    print('Starting Decrypto Networko')
    
    ser = server.Server(3, 7)
    cli = client.Client(3, 7)

    data = 'Hi there my name is Arko and I am here to study computer science and engineering'

    print('------------------------Phase 1 -------------------------')
    print('Encrypting!')
    encrypted_data = ser.encrypt(data)
    print(encrypted_data)

    print('Decrypting')
    decrypted_data = cli.decrypt(encrypted_data)
    print(decrypted_data)

    print('------------------------Phase 2 -------------------------')
    data = 'Testing'
    print('Encrypting!')
    encrypted_data = ser.encrypt(data)

    print('Decrypting')
    decrypted_data = cli.decrypt(encrypted_data)
    print(decrypted_data)

    print('------------------------Phase 3 -------------------------')
    data = 'This is another set of data that needs to be encrypted and sent to the client'
    print('Encrypting!')
    encrypted_data = ser.encrypt(data)

    print('Decrypting')
    decrypted_data = cli.decrypt(encrypted_data)
    print(decrypted_data)
