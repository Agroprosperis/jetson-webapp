ACCESS_COOKIE_NAME = "tilletia_access_token"
REFRESH_COOKIE_NAME = "tilletia_refresh_token"


def clear_auth_cookies(response):
    response.delete_cookie(ACCESS_COOKIE_NAME)
    response.delete_cookie(REFRESH_COOKIE_NAME)
    return response


def get_access_token_from_request(request):
    return (request.cookies.get(ACCESS_COOKIE_NAME) or "").strip()


def get_refresh_token_from_request(request):
    return (request.cookies.get(REFRESH_COOKIE_NAME) or "").strip()


def set_auth_cookies(response, access_token, refresh_token, *, access_max_age, refresh_max_age):
    response.set_cookie(
        ACCESS_COOKIE_NAME,
        access_token,
        max_age=access_max_age,
        httponly=True,
        samesite="Lax",
    )
    response.set_cookie(
        REFRESH_COOKIE_NAME,
        refresh_token,
        max_age=refresh_max_age,
        httponly=True,
        samesite="Lax",
    )
    return response
