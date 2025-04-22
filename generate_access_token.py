from kiteconnect import KiteConnect

api_key = "bwhraj28ii33624u"
api_secret = "2926p2qcpjcb61aectwo9z2m24bb6x8y"
request_token = "Au452VB5vLuMzM5gsT4eHP2EjW8fmcxj"

kite = KiteConnect(api_key=api_key)
data = kite.generate_session(request_token, api_secret=api_secret)
print("✅ Access Token:", data["access_token"])