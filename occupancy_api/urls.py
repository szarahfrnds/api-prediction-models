from django.contrib import admin
from django.urls import path, include

# --- NOVAS IMPORTAÇÕES DO SWAGGER ---
from rest_framework import permissions
from drf_yasg.views import get_schema_view
from drf_yasg import openapi
# ----------------------------------------------------
# 1. CONFIGURAÇÃO DO SWAGGER/OPENAPI
# ----------------------------------------------------
schema_view = get_schema_view(
   openapi.Info(
      title="API de Predição de Modelos (Docs)",
      default_version='v1',
      description="Documentação da sua API Django.",
      terms_of_service="https://www.google.com/policies/terms/",
      contact=openapi.Contact(email="saraoliveirafernandes3@gmail.com"), 
      license=openapi.License(name="BSD License"),
   ),
   public=True,
   permission_classes=(permissions.AllowAny,),
)
# ----------------------------------------------------


urlpatterns = [
    # Suas rotas existentes:
    path("admin/", admin.site.urls),
    path("api/", include("predictions.urls")), 
    
    # ----------------------------------------------------
    # 2. INCLUSÃO DAS ROTAS DO SWAGGER:
    # ----------------------------------------------------
    # Esta é a rota que seu dev sênior quer: /swagger/
    path('swagger/', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    
    # Esta é uma rota alternativa, mais limpa (Redoc):
    path('redoc/', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),
    
    # A rota para o arquivo de especificação (JSON)
    path('swagger.json/', schema_view.without_ui(cache_timeout=0), name='schema-json'),
    # ----------------------------------------------------
]