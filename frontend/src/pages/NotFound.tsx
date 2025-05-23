import { useLocation } from 'react-router-dom'
import { useEffect } from 'react'

const NotFound = () => {
  const location = useLocation()

  useEffect(() => {
    console.error(
      '404 Error: Usuario intentó acceder a una ruta inexistente',
      location.pathname
    )
  }, [location.pathname])

  return (
    <div className="flex min-h-screen items-center justify-center">
      <div className="text-center">
        <h1 className="mb-4 text-4xl font-bold">404</h1>
        <p className="mb-4 text-xl">Ocurrió un error. Página no encontrada.</p>
        <a href="/" className="text-blue-500 underline hover:text-blue-700">
          Volver al inicio
        </a>
      </div>
    </div>
  )
}

export default NotFound
