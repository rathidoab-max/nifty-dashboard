from kiteconnect import KiteConnect

api_key = "bwhraj28ii33624u"
access_token = "3KZ7HAYfKjA13lToH18kkPc7os4W2FiM"

kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

print(kite.profile())