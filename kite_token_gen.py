from kiteconnect import KiteConnect

api_key = "bwhraj28ii33624u"
api_secret = "2926p2qcpjcb61aectwo9z2m24bb6x8y"

kite = KiteConnect(api_key=api_key)
print("🔐 Login URL:", kite.login_url())